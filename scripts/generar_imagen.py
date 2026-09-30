#!/usr/bin/env python3
"""Genera imagenes en local con Qwen-Image-2.1 (pesos abiertos, cuantizados GGUF).

Qwen-Image-2.1 no tiene API alojada: se ejecuta con diffusers sobre los pesos
descargados de HuggingFace. El denoiser (transformer) se carga cuantizado en
GGUF y el text encoder Qwen3-VL 8B se queda en la RAM: en bf16 son ~17 GB y no
caben en una GPU de 8 GB, asi que se codifica el prompt en CPU y solo los
embeddings viajan a la GPU.

Tres detalles que no son obvios y que rompen la carga si se ignoran:

1. Los checkpoints GGUF no se pueden pasar por `Pipeline.from_pretrained`.
   Diffusers solo los acepta via `from_single_file` sobre el transformer, que
   despues se inyecta en el pipeline con el kwarg `transformer=`. El `config`
   que exige `from_single_file` es la carpeta `base/transformer/`.
2. `compute_dtype` es un `torch.dtype`, no una cadena: el cuantizador lo usa
   como destino de `.to()` al dequantizar, y `"bfloat16"` revienta con
   "Invalid device string".
3. Los pesos 1-D de las RMSNorm (`norm_q`, `norm_k`, `text_norm`) tambien
   vienen cuantizados, pero no son `Linear`, asi que el cuantizador GGUF no los
   envuelve y el forward multiplica por bytes crudos ("size of tensor a (4096)
   must match tensor b (8192)"). `modules_to_not_convert` los dequantiza al
   cargar. Sin esto, revienta en el primer bloque.

El decode del VAE va en tiles: a 1024x1024 la activacion del decoder no cabe en
la VRAM que deja libre el denoiser.

Disposicion de pesos esperada (por defecto en ~/qwen-weights):
    base/
      model_index.json
      processor/  scheduler/  vae/  text_encoder/
      transformer/
        config.json
    transformer-gguf/
      qwen-image-2.1-Q4_K_M.gguf

Las rutas de salida son relativas a la raiz del repo, no al directorio desde el que
se invoque el script: asi `salida/Imagen.png` cae siempre en MinAgent/salida/, aunque
se llame desde /tmp o desde otro proyecto.

Ejemplo:
    python scripts/generar_imagen.py "un ferrari rojo"
    python scripts/generar_imagen.py "un gato" --width 2048 --height 2048 --steps 40
    python scripts/generar_imagen.py "un sticker de dragon" --transparente
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from pathlib import Path

# Fragmentation, not capacity, is what usually kills this run: the denoiser
# allocates and frees in bursts and the default allocator leaves hundreds of MiB
# reserved but unusable, so a request for 96 MiB fails on a card that looks half
# empty. This has to be set before CUDA initialises, which is why it is here and
# not next to the pipeline. setdefault: an explicit value in the environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_BASE = Path.home() / "qwen-weights" / "base"
DEFAULT_GGUF = Path.home() / "qwen-weights" / "transformer-gguf"

# Raiz del repo (este script vive en scripts/). Las rutas relativas se anclan aqui
# y no al cwd, para que la salida no dependa de donde se ejecute.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "salida"

# Solo se acepta un unico cuant por modelo: el denoiser se carga con
# from_single_file, que no admite checkpoints troceados.
MAX_GGUF_FILES = 1

# Resoluciones nativas soportadas por Qwen-Image-2.1 (2K).
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

PREFIX_RGBA = (
    "This is an RGBA image with transparency. {description}. "
    "The image has alpha channel and the background is transparent."
)

VENV_PYTHON = Path.home() / "qwenimg-venv" / "bin" / "python"

# Modulos de RMSNorm: sus pesos son 1-D y el cuantizador GGUF solo sabe
# envolver Linear, asi que hay que excluirlos para que se dequanticen.
UNQUANTIZED_MODULES = ["norm_q", "norm_k", "text_norm"]

# Lado del tile del decoder del VAE, en pixeles de latente.
VAE_TILE_SIZE = 256


def slugify(text: str, max_words: int = 6) -> str:
    """Convierte un prompt en un nombre de fichero seguro y legible."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    words = [w for w in re.split(r"[\s_-]+", text) if w][:max_words]
    return "-".join(words) or "imagen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera imagenes en local con Qwen-Image-2.1 (GGUF cuantizado).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", help="Descripcion de la imagen a generar.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guarda la imagen. Las rutas relativas se resuelven "
             "contra la raiz del repo, no contra el directorio actual.",
    )
    parser.add_argument(
        "-n", "--name", default=None,
        help="Nombre del fichero sin extension. Por defecto se deriva del prompt.",
    )
    parser.add_argument(
        "--aspect", choices=sorted(ASPECT_RATIOS), default=None,
        help="Relacion de aspecto nativa. Si se indica, ignora --width/--height.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Ancho en pixeles.")
    parser.add_argument("--height", type=int, default=1024, help="Alto en pixeles.")
    parser.add_argument("--steps", type=int, default=30, help="Pasos de difusion.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria.")
    parser.add_argument(
        "--cfg-scale", type=float, default=1.0,
        help="Classifier-free guidance. Qwen-Image 2.1 esta entrenado sin guidance, "
             "asi que 1.0 (sin negative prompt) es el valor recomendado.",
    )
    parser.add_argument(
        "--negative-prompt", default="low resolution, blurry, distorted, watermark, text",
        help="Prompt negativo. Solo se usa si --cfg-scale es mayor que 1.",
    )
    parser.add_argument(
        "--transparente", action="store_true",
        help="Genera una imagen RGBA con fondo transparente.",
    )
    parser.add_argument(
        "--text-encoder-device", choices=["cpu", "auto"], default="cpu",
        help="Donde vive el text encoder de 8B. 'cpu' cabe en cualquier GPU; "
             "'auto' lo mete en la GPU y exige ~17GB de VRAM libres.",
    )
    parser.add_argument(
        "--vae-tile", type=int, default=VAE_TILE_SIZE,
        help="Lado del tile del decoder del VAE en latentes. 0 desactiva el tiling.",
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE, help="Carpeta de los pesos base.")
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF, help="Carpeta con el .gguf cuantizado.")
    return parser.parse_args(argv)


def find_gguf(gguf_dir: Path) -> Path:
    """Devuelve el unico .gguf del denoiser.

    No hay enlaces simbolicos en base/transformer/: el GGUF se carga por ruta
    absoluta desde aqui, asi que no hace falta tocar el arbol de pesos base.
    """
    candidates = sorted(gguf_dir.glob("*.gguf"))
    if not candidates:
        raise SystemExit(f"No se encuentra ningun .gguf en {gguf_dir}")
    if len(candidates) > MAX_GGUF_FILES:
        names = ", ".join(c.name for c in candidates)
        raise SystemExit(
            f"Se esperaba un solo .gguf en {gguf_dir} y hay {len(candidates)}: {names}. "
            "Deja solo el cuant que quieras usar o pasalo con --gguf-dir."
        )
    return candidates[0]


def check_weights(base_dir: Path, gguf_dir: Path) -> None:
    missing = []
    for path, label in ((base_dir / "model_index.json", "model_index.json"),
                        (base_dir / "text_encoder", "text_encoder/"),
                        (base_dir / "vae", "vae/"),
                        (base_dir / "processor", "processor/"),
                        (base_dir / "scheduler", "scheduler/"),
                        (base_dir / "transformer" / "config.json", "transformer/config.json"),
                        (gguf_dir, "gguf-dir/")):
        if not path.exists():
            missing.append(f"{label} (en {path})")
    if missing:
        raise SystemExit(
            "Faltan pesos de Qwen-Image-2.1:\n  - "
            + "\n  - ".join(missing)
            + "\n\nDescargalos con el script de instalacion de la skill crear-imagen-qwen."
        )


def resolve_output_dir(output_dir: Path) -> Path:
    """Devuelve la carpeta de salida absoluta, anclada a la raiz del repo.

    Una ruta relativa se resuelve contra REPO_ROOT y no contra el cwd, para que
    `salida` signifique lo mismo se ejecute el script desde donde se ejecute.
    """
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT)
    except ValueError:
        raise SystemExit(
            f"El directorio de salida {output_dir} esta fuera del repo ({REPO_ROOT}). "
            "Las imagenes del proyecto van en salida/."
        ) from None
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_weights(args.base_dir, args.gguf_dir)
    gguf_path = find_gguf(args.gguf_dir)
    print(f"Denoiser cuantizado: {gguf_path.name}")

    try:
        import torch
        from diffusers import (
            GGUFQuantizationConfig,
            QwenImage21Pipeline,
            QwenImage21Transformer2DModel,
        )
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise SystemExit(
            f"Falta '{exc.name}'. Instala las dependencias con el venv de la skill "
            f"crear-imagen-qwen ({VENV_PYTHON})."
        ) from exc

    if args.aspect:
        width, height = ASPECT_RATIOS[args.aspect]
    else:
        width, height = args.width, args.height
    if width % 16 or height % 16:
        width += 16 - (width % 16)
        height += 16 - (height % 16)
        print(f"Ajustado a multiplos de 16: {width}x{height}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: no se detecta CUDA; la generacion en CPU sera muy lenta.")

    prompt = PREFIX_RGBA.format(description=args.prompt) if args.transparente else args.prompt

    print(f"Cargando Qwen-Image-2.1 (GGUF) en {device}...")
    started = time.perf_counter()
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    # Sin esto los pesos 1-D de las RMSNorm se quedan como bytes cuantizados.
    quantization_config.modules_to_not_convert = list(UNQUANTIZED_MODULES)
    transformer = QwenImage21Transformer2DModel.from_single_file(
        str(gguf_path),
        config=str(args.base_dir / "transformer"),
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        str(args.base_dir),
        transformer=transformer,
        dtype=torch.bfloat16,
    )
    if args.vae_tile:
        pipe.vae.enable_tiling(args.vae_tile, args.vae_tile)
    pipe.set_progress_bar_config(disable=True)

    if device == "cpu":
        pipe.to("cpu")
    else:
        encoder_device = device if args.text_encoder_device == "auto" else "cpu"
        pipe.transformer.to(device)
        pipe.vae.to(device)
        pipe.text_encoder.to(encoder_device)
        if encoder_device == "cpu":
            print("Text encoder (8B) en CPU: se codifica el prompt ahi y solo viajan los embeddings a la GPU.")
    print(f"Modelo cargado en {time.perf_counter() - started:.1f}s")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # El prompt se codifica aparte y se pasa como prompt_embeds: el pipeline
    # asumiiria el device de ejecucion (la GPU) y el text encoder esta en CPU.
    started = time.perf_counter()
    prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(prompt, device=encoder_device)
    negative_embeds = negative_embeds_mask = None
    if args.cfg_scale > 1:
        negative_embeds, negative_embeds_mask, _ = pipe.encode_prompt(
            args.negative_prompt, device=encoder_device
        )
        if device != "cpu":
            negative_embeds = negative_embeds.to(device)
    if device != "cpu":
        prompt_embeds = prompt_embeds.to(device)
    print(f"Prompt codificado en {time.perf_counter() - started:.1f}s")

    print(f"Generando {width}x{height}, {args.steps} pasos...")
    started = time.perf_counter()
    result = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_embeds,
        negative_prompt_embeds_mask=negative_embeds_mask,
        true_cfg_scale=args.cfg_scale,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        generator=generator,
    )
    elapsed = time.perf_counter() - started

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or slugify(args.prompt)
    path = output_dir / f"{name}.png"
    counter = 2
    while path.exists():
        path = output_dir / f"{name}-{counter}.png"
        counter += 1

    image = result.images[0]
    if args.transparente and image.mode != "RGBA":
        image = image.convert("RGBA")
    image.save(path)

    print(f"Imagen guardada en {path.relative_to(REPO_ROOT)} ({image.mode}, "
          f"{image.width}x{image.height})")
    print(f"Tiempo de generacion: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

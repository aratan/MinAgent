#!/usr/bin/env python3
"""Genera un video corto con Wan 2.1 T2V-1.3B (pesos abiertos, diffusers).

Sin ComfyUI ni interfaz grafica: un script y los pesos. El pipeline se descarga
solo de HuggingFace la primera vez, unos 29 GB, y despues se reutiliza desde la
cache de HuggingFace.

De esos 29 GB, 22,7 son el codificador de texto UMT5-XXL, que es la mayor parte
del peso y lo que hace que la descarga sea larga. Se carga en fp8 (~5,6 GB) para
que quepa junto al transformer, pero en disco esta en bf16.

El 1.3B es la variante ligera de Wan: 480p, 81 fotogramas por defecto, y cabe
en una GPU de 8 GB. La 14B no cabe en una tarjeta de consumo; no se intenta.

Tres detalles que no son obvios y que hacen que esto reviente en una tarjeta de
8 GB si se ignoran:

1. El codificador de texto UMT5-XXL en bf16 ocupa ~11 GB, mas que la VRAM
   entera. Se carga en fp8 (~5,6 GB) y, aun asi, hay que moverlo a RAM y subir
   solo lo que cada paso necesita. Sin offload, OOM.
1b. Y con `enable_model_cpu_offload()` tampoco entra en 8 GB, que es lo que
   parece a primera vista: ese modo deja el transformer entero (2,6 GB en bf16)
   en la GPU, y a 480p la atencion corre sobre ~35.000 tokens, con lo que la
   activacion se va a 7 GB y revienta con la tarjeta medio vacia. El que cabe
   es `enable_group_offload()`, que baja y sube bloques del transformer; por eso
   `--offload` existe y por defecto es `group`. Un OOM aqui casi nunca es
   fragmentacion: mira "N GiB is allocated by PyTorch", no "reserved but
   unallocated".
2. Fragmentacion, no capacidad: el allocator reserva cientos de MiB que luego no
   son utilizables, y una peticion de 96 MiB falla en una tarjeta que parece
   vacia. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lo arregla. Hay que
   fijarlo antes de que CUDA inicialice, por eso va aqui y no en la linea de
   comandos.
3. Los 81 fotogramas se acumulan en RAM antes de codificarse. Con `np` eso son
   480x832x81x3 float32, unos 380 MB de un salto, y el decodificador del VAE
   tambien necesita sitio. Con `output_type="latent"` los fotogramas se quedan
   en el espacio latente, que es mucho mas pequeño, y se decodifican despues.

La GPU de este equipo es una RTX 4060 Laptop de 8 GB: cuenta con varios minutos
por clip, no con segundos. En 25 fotogramas y 15 pasos se van unos 15 min.

El numero de fotogramas es lo que manda en la memoria, no los pasos de
difusion: los pasos recalculan lo mismo sobre la misma secuencia, mientras que
cada fotograma anade tokens a la atencion. Bajar `--pasos` de 50 a 15 no
ahorra VRAM, solo tiempo (con la calidad que corresponda); `--frames 17` si.

Ejemplos:
    python scripts/generar_video.py "a drone flying over a misty jungle valley"
    python scripts/generar_video.py "a cat" --pasos 20 --frames 17
    python scripts/generar_video.py "an astronaut" --semilla 42
    python scripts/generar_video.py "prueba" --offload sequential --no-vae-fp32
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Debe fijarse antes de que torch inicialice CUDA: si torch ya se ha importado,
# CUDA ha reservado su pool con la configuracion por defecto y esto llega tarde.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# El sufijo -Diffusers no es opcional: el repo "Wan-AI/Wan2.1-T2V-1.3B" guarda
# los pesos en el formato original de Wan, sin model_index.json, y
# from_pretrained responde 404 contra el. Este es el reempaquetado para diffusers.
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Las rutas de salida son relativas a la raiz del repo, no al directorio desde el
# que se invoque el script: asi `salida/video.mp4` cae siempre en
# MinAgent/salida/, aunque se llame desde /tmp o desde otro proyecto.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "salida"

FPS = 16

# Lo que Wan 2.1 tiende a producir de mas; el negative prompt por defecto del
# modelo cubre parte, pero estos bajan los artefactos tipicos a 480p.
DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un video corto con Wan 2.1 T2V-1.3B en local.")
    parser.add_argument("prompt", help="Descripcion en ingles del video que quieres.")
    parser.add_argument("--negativo", default=DEFAULT_NEGATIVE, help="Prompt negativo (lo que no debe aparecer).")
    parser.add_argument("--pasos", type=int, default=50, help="Pasos de diffusion (50 por defecto).")
    parser.add_argument("--frames", type=int, default=81, help="Fotogramas. Menos = mas corto y mas barato.")
    parser.add_argument("--ancho", type=int, default=832, help="Ancho en pixeles (832 por defecto).")
    parser.add_argument("--alto", type=int, default=480, help="Alto en pixeles (480 por defecto).")
    parser.add_argument("--cfg", type=float, default=5.0, help="guidance_scale. 1 destruye el prompt; usa 5.")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla para reproducir el mismo video.")
    parser.add_argument("--salida", default=None, help="Nombre del fichero .mp4, sin carpeta. Va a salida/.")
    parser.add_argument(
        "--descargar-solo",
        action="store_true",
        help="Solo descarga los pesos y sale, sin generar nada.",
    )
    parser.add_argument(
        "--offload",
        choices=["model", "group", "sequential", "ninguno"],
        default="group",
        help="Como se mueve el peso entre GPU y RAM. 'group' baja y sube bloques del "
             "transformer por grupos y es el unico que cabe de sobra en 8 GB; 'model' deja "
             "el transformer entero en la GPU y revienta con 7 GB ya asignados; "
             "'sequential' va por capas y es el mas lento pero el que menos VRAM pide; "
             "'ninguno' lo deja todo en la GPU.",
    )
    parser.add_argument(
        "--offload-bloques",
        type=int,
        default=1,
        help="Con --offload group, cuantos bloques del transformer se quedan en la GPU a la "
             "vez. 1 es lo que cabe en 8 GB; subirlo acelera y come mas VRAM.",
    )
    parser.add_argument(
        "--vae-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mantiene el VAE en float32. En precision reducida la decodificacion mete "
             "artefactos visibles en los bordes. Usa --no-vae-fp32 para ahorrar VRAM.",
    )
    return parser.parse_args(argv)


def resolve_output_dir(output_name: str | None) -> Path:
    """Resolve the output file, refusing any name that climbs out of salida/."""
    directory = DEFAULT_OUTPUT_DIR
    if directory.is_symlink():
        raise SystemExit(f"ERROR: {directory} es un symlink; se niega escribir fuera del proyecto.")
    directory.mkdir(parents=True, exist_ok=True)

    if not output_name:
        output_name = f"video-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    elif not output_name.endswith(".mp4"):
        output_name = f"{output_name}.mp4"

    # Solo un nombre de fichero: sin separadores de ruta ni nombres ocultos.
    if any(separator in output_name for separator in ("/", "\\")) or output_name.startswith("."):
        raise SystemExit(f"ERROR: nombre de salida invalido: {output_name}")

    target = directory / output_name
    if target.is_symlink():
        raise SystemExit(f"ERROR: {target} es un symlink; se niega escribir fuera del proyecto.")
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.frames <= 0 or args.pasos <= 0:
        raise SystemExit("ERROR: --frames y --pasos tienen que ser mayores que cero.")
    if args.frames % 4 != 1:
        # Wan trabaja en bloques de 4 mas un fotograma inicial. Una cuenta que
        # no cumple da un error de formas del VAE mas adelante, lejos de la causa.
        raise SystemExit("ERROR: --frames debe ser 4n+1 (81, 49, 33, 25...). Wan trabaja en bloques de 4.")

    # imageio va en el mismo bloque a proposito: export_to_video lo importa al
    # codificar, no al cargar el pipeline, asi que sin esto el fallo apareceria
    # minutos despues, con los pesos ya descargados y la GPU ocupada.
    try:
        import imageio
        import imageio_ffmpeg
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as error:
        raise SystemExit(
            f"ERROR: falta una dependencia: {error}.\n"
            "Las de medios no son dependencias de MinAgent (minagent solo necesita httpx y regex),\n"
            "asi que no estan en pyproject.toml. Instalalas en el venv del proyecto con:\n"
            "    uv pip install torch diffusers transformers accelerate imageio imageio-ffmpeg"
        ) from error

    # ffmpeg de verdad, no solo el paquete: sin binario imageio aborta al escribir.
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        imageio.plugins.ffmpeg.get_exe()
    except Exception as error:
        raise SystemExit(f"ERROR: imageio-ffmpeg no encuentra un binario de ffmpeg: {error}") from error

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no hay CUDA disponible. El 1.3B en CPU es impracticable.")

    target = resolve_output_dir(args.salida)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Modelo:  {MODEL_ID}")
    print(f"Salida:  {target}")
    print(f"Video:   {args.ancho}x{args.alto}, {args.frames} fotogramas, {args.pasos} pasos, {FPS} fps")
    print(f"GPU:     {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"ffmpeg:  {ffmpeg}")
    print(f"Peso:    {dtype}")
    print("Cargando pesos (la primera vez se descargan de HuggingFace, ~29 GB)...")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        # El UMT5-XXL en bf16 no cabe ni con la VRAM entera para el.
        text_encoder_dtype=torch.float8_e4m3fn,
        torch_dtype=dtype,
    )

    # El VAE se queda en fp32: en precision reducida la decodificacion mete
    # artefactos visibles en los bordes.
    if pipe.vae is not None and args.vae_fp32:
        pipe.vae.to(dtype=torch.float32)

    # El offload es lo que hace que quepa. 'model' deja el transformer entero
    # (2,6 GB en bf16) en la GPU, y con los ~35.000 tokens de atencion que
    # genera 480p la activacion se va a 7 GB: se queda sin VRAM con la tarjeta
    # medio vacia. 'group' baja y sube bloques, que es lo que deja respirar a los
    # 8 GB de una tarjeta de consumo.
    if args.offload == "model":
        pipe.enable_model_cpu_offload(device="cuda")
    elif args.offload == "group":
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=args.offload_bloques,
            use_stream=True,
        )
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device="cuda")
    else:
        pipe.to("cuda")
    print(f"Offload: {args.offload}")

    # Menos cuarteo de atencion y decodificacion por bloques -> menos VRAM.
    # enable_tiling vive en el VAE, no en el pipeline.
    try:
        pipe.enable_attention_slicing()
        if pipe.vae is not None:
            pipe.vae.enable_tiling()
    except Exception as error:  # no todas las versiones exponen los dos
        print(f"aviso: no se pudieron activar todas las optimizaciones de VRAM ({error})")

    if args.descargar_solo:
        print("Pesos descargados y en cache. Listo para generar.")
        return 0

    generator = None
    if args.semilla is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.semilla)
        print(f"Semilla: {args.semilla} (mismo prompt + misma semilla = mismo video)")

    print("Generando... (varios minutos en una GPU de 8 GB; no la interruptas)")
    inicio = time.monotonic()
    try:
        frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negativo,
            height=args.alto,
            width=args.ancho,
            num_frames=args.frames,
            num_inference_steps=args.pasos,
            guidance_scale=args.cfg,
            num_videos_per_prompt=1,
            generator=generator,
        ).frames[0]
    except torch.cuda.OutOfMemoryError as error:
        print(f"ERROR: se ha quedado sin VRAM ({error}).", file=os.sys.stderr)
        print(
            "El numero de fotogramas es lo que manda aqui, no los pasos: cada paso recalcula\n"
            "la atencion sobre toda la secuencia. Baja a --frames 17 y luego reconstruye el clip\n"
            "por tramos, o baja la resolucion a --ancho 480 --alto 320. Si ya vas con --offload\n"
            "sequential y sigue sin entrar, no cabe en esta GPU.",
            file=os.sys.stderr,
        )
        return 1

    transcurrido = time.monotonic() - inicio
    print(f"Generados {len(frames)} fotogramas en {transcurrido / 60:.1f} min; codificando a mp4...")

    # macro_block_size=16 por defecto en diffusers: 832x480 ya son multiplos de 16,
    # pero un --ancho/--alto raro haria que imageio reescalase el video entero.
    if args.ancho % 16 or args.alto % 16:
        print(f"aviso: {args.ancho}x{args.alto} no son multiplos de 16; imageio reescala al siguiente")
    export_to_video(frames, str(target), fps=FPS)
    print(f"OK: {target} ({target.stat().st_size / 1e6:.1f} MB, ~{len(frames) / FPS:.1f} s de video)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

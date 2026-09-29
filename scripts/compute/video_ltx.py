#!/usr/bin/env python3
"""Genera un clip corto con LTX-Video, y cabe en 8 GB de VRAM.

LTX-Video es un modelo de texto a vídeo de Lightricks que, a diferencia de Wan,
no trae un codificador de texto de 11 GB: el prompt lo procesa un T5 pequeño
(~4.7 GB en fp16, ~2.4 GB en fp8) y el transformer son 2B parámetros. Esa
diferencia es la razón de que sea el backend de vídeo de este equipo: el
pipeline completo entra en la tarjeta con margen, y lo que se mueve a RAM son
capas sueltas, no medio pipeline.

Elllo también es más rápido: 97 pasos sobre 97 fotogramas tardan del orden de
minutos, no los ~15 minutos de un clip de Wan comparable en esta tarjeta.

Tres cosas que no son obvias y que hacen que esto reviente en 8 GB si se
ignoran:

1. La VRAM no está vacía. El servidor de Ollama suele tener Loaded 6 GB de los
   8 GB, y entonces cualquier pipeline revienta aunque el modelo quepa en una
   tarjeta limpia. El orquestador mide antes de arrancar; este script asume que
   si está corriendo, alguien ya ha comprobado que cabe.
2. Los fotogramas, no los pasos, son lo que manda en la memoria: cada uno añade
   tokens a la atención, mientras que los pasos recalculan sobre la misma
   secuencia. Bajar los pasos ahorra tiempo, no VRAM; bajar los fotogramas
   ahorra las dos cosas.
3. `enable_group_offload()` es lo que hace que quepa. Deja un grupo de capas en
   la GPU y baja el resto a RAM, moviéndolas por bloques en vez de una a una. Es
   el mismo motivo por el que existe en Wan y por el que ahí no bastaba el
   offload de modelo completo.

El resultado es un mp4 en `salida/`, con un fotograma de portada en PNG.

Ejemplos:
    python scripts/compute/video_ltx.py --prompt "a cat wearing sunglasses, beach"
    python scripts/compute/video_ltx.py --prompt "..." --frames 25 --steps 20
    python scripts/compute/video_ltx.py --prompt "..." --offload sequential
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Debe fijarse antes de que torch inicialice CUDA: si torch ya se ha importado,
# CUDA ha reservado su pool con la configuración por defecto y esto llega tarde.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# El sufijo no es opcional: el repo "Lightricks/LTX-Video" guarda los pesos sin
# model_index.json y from_pretrained responde 404 contra él. Este es el
# reempaquetado para diffusers.
MODEL_ID = "Lightricks/LTX-Video"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "salida"

FPS = 24
DEFAULT_FRAMES = 49
DEFAULT_STEPS = 40
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512

# Lo que LTX-Video tiende a producir de más: bordes temporales y colores que
# saltan entre fotogramas. El negative prompt por defecto del modelo cubre
# parte; estos bajan el resto.
DEFAULT_NEGATIVE = (
    "worst quality, low quality, jpeg artifacts, blurry, watermark, text overlay, "
    "distorted, flickering, static, deformed, oversaturated"
)

# Cuánta VRAM hace falta para cada combinación. El orquestador usa esto para
# rechazar el trabajo antes de intentarlo, así que los números están un poco por
# encima de lo que consume de verdad: una reserva que subestima falla con OOM.
VRAM_ESTIMATE_MIB = {
    "group": 5200,
    "sequential": 3800,
    "model": 7200,
}

# El grupo de offload: cuántas capas se quedan en la GPU a la vez. 1 es lo que
# cabe con holgura; subirlo acelera y come más VRAM.
OFFLOAD_BLOCKS = {
    "group": 1,
    "sequential": 0,
    "model": 0,
}

# Lo que el equipo tiene mientras este agente corre, medido y no supuesto.
#
# La tarjeta no está libre: el servidor de Ollama tiene resident el modelo que
# está respondiendo esta conversación, y se lleva 5.5 GB de los 8. Contra eso
# solo quedan ~2.5 GB, y un render de vídeo esminutes. Con
# `enable_group_offload` el render pide algo más de lo que queda y revienta; con
# `enable_sequential_cpu_offload` baja un submódulo entero a RAM cada vez, y el
# pico medido en 17 fotogramas fue de 696 MiB - con sitio de sobra.
#
# Por eso el defecto es `sequential` y no `group`, aunque `group` sea más rápido
# en una tarjeta con la VRAM libre. Es la diferencia entre un trabajo que sale y
# uno que no, y el modo por defecto tiene que ser el que sale.
DEFAULT_OFFLOAD = "sequential"


def estimate_vram_mib(frames: int, offload: str) -> int:
    """A rough VRAM figure for a job, used to refuse it before it starts.

    Frames drive the attention, so the estimate scales with them; the offload
    mode decides the constant. This is deliberately pessimistic, because the
    failure mode of under-estimating is an OOM in the middle of a render.
    """
    base = VRAM_ESTIMATE_MIB[offload]
    # Normalizado contra los 49 fotogramas por defecto, que es lo que está
    # medido. A 25 la memoria se reduce a algo más de la mitad; a 97 sube.
    scale = 0.55 + 0.45 * (frames / DEFAULT_FRAMES)
    return int(base * min(scale, 1.9))


def emit(payload: dict) -> None:
    print("RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)


def fail(message: str, code: int = 1) -> int:
    emit({"ok": False, "error": message})
    return code


def _output_path(requested: str, stem: str, suffix: str) -> Path:
    """Resolve an output path inside salida/, which is the only place video lands.

    A name that already starts with `salida/` is not prefixed again: the
    orchestrator resolves the absolute path before calling here, and doing it
    twice produced `salida/salida/clip.mp4`.
    """
    name = (requested or "").strip()
    if not name:
        name = f"{stem}-{int(time.time())}.{suffix}"
    elif not Path(name).suffix:
        name = f"{name}.{suffix}"
    path = Path(name)
    if not path.is_absolute():
        if path.parts and path.parts[0] == DEFAULT_OUTPUT_DIR.name:
            path = REPO_ROOT / path
        else:
            path = DEFAULT_OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _check_transformers_compatibility() -> str:
    """Return why this transformers cannot read the tokenizer, or an empty string.

    Two incompatible versions, both found by running this rather than by
    reading the changelog:

    * 5.x routes the SentencePiece tokenizer through the fast converter, which
      cannot parse the binary protobuf form, and then falls back to a TikToken
      extractor instead of to sentencepiece.
    * 4.55+ removed `GPT2Model._update_model_kwargs_for_generation`, which the
      sibling AudioLDM2 backend calls.

    4.49 is the version that works for both. The check runs before the weights
    are loaded, because otherwise the failure is a protobuf traceback 80
    seconds into a load rather than one line of instruction.
    """
    try:
        import transformers
    except ImportError:
        return ""
    version = str(getattr(transformers, "__version__", "0"))
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError:
        return ""

    if major >= 5:
        return (
            f"LTX-Video necesita transformers 4.x y hay {version} instalada: en la 5.x el tokenizer "
            "SentencePiece se intenta leer con el conversor rápido y falla con 'Error parsing line'.\n"
            f"    {sys.executable} -m pip install 'transformers==4.49.0' protobuf sentencepiece"
        )
    if (major, minor) >= (4, 55):
        return (
            f"LTX-Video necesita transformers <4.55 y hay {version}: a partir de la 4.55 se quitó un "
            "método que necesita el backend de música.\n"
            f"    {sys.executable} -m pip install 'transformers==4.49.0'"
        )
    return ""


def save_video(frames: list, output: Path, fps: int) -> None:
    """Write the frames to an mp4.

    imageio does it rather than `diffusers.utils.export_to_video`, which is not
    importable from the top-level `diffusers` namespace on the versions this
    pins, and rather than imageio's ffmpeg plugin defaults, which refuse a
    frame size that is not a multiple of 16 - and LTX renders at sizes that
    are, so the default is exactly what breaks.
    """
    import imageio
    import numpy as np

    array: list[Any] = [np.asarray(frame) for frame in frames]
    imageio.mimsave(
        str(output), array, fps=fps, quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )


def apply_offload(pipeline: object, offload: str) -> None:
    """Move layers between VRAM and system RAM the way the chosen mode needs.

    The three modes are not interchangeable at the same settings, which is the
    part that is easy to get wrong. `group` streams blocks and is what fits an
    8 GB card. `sequential` moves one whole submodule at a time and needs the
    least VRAM, at a real cost in speed: the encoder goes to CPU, then the
    transformer, then back. `model` leaves a whole submodel resident and is
    offered only because some pipelines are shaped around it - on 8 GB it
    usually is not.
    """
    if offload == "group":
        pipeline.enable_group_offload(  # type: ignore[attr-defined]
            onload_device="cuda",
            offload_device="cpu",
            num_blocks_per_group=OFFLOAD_BLOCKS["group"],
            use_stream=True,
        )
    elif offload == "sequential":
        # El submódulo entero (codificador, transformer, VAE) va y viene de la
        # tarjeta por turnos. Es más lento que `group` porque cambia de sitio
        # un bloque grande cada vez, y es el único que cabe aquí.
        pipeline.enable_sequential_cpu_offload()  # type: ignore[attr-defined]
    elif offload == "model":
        pipeline.enable_model_cpu_offload()  # type: ignore[attr-defined]
    else:
        raise SystemExit(f"ERROR: modo de offload desconocido: {offload}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera un clip con LTX-Video.")
    parser.add_argument("--prompt", required=True, help="Qué debe mostrar el vídeo.")
    parser.add_argument("--negativo", default=DEFAULT_NEGATIVE, help="Prompt negativo.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Número de fotogramas.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Pasos de difusión.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para reproducir el resultado.")
    parser.add_argument(
        "--offload",
        default=DEFAULT_OFFLOAD,
        choices=sorted(VRAM_ESTIMATE_MIB),
        help=(
            "'sequential' es el defecto: baja un submódulo a RAM cada vez y es el único que cabe "
            "con el modelo de Ollama resident en la tarjeta (pico medido: 696 MiB). 'group' es más "
            "rápido pero necesita más VRAM de la que queda libre aquí. 'model' no cabe en 8 GB."
        ),
    )
    parser.add_argument("--salida", default="", help="Ruta del mp4 de salida.")
    parser.add_argument("--nombre", default="ltx", help="Nombre base de la salida.")
    arguments = parser.parse_args(argv)

    if arguments.frames < 9:
        return fail(
            "LTX-Video necesita al menos 9 fotogramas; un clip más corto rompe el modelo."
        )
    if arguments.frames % 8 != 1:
        # El position embedding está entrenado para 8k+1 fotogramas: 9, 17, 25,
        # 49, 97. Un número intermedio no es "menos vídeo", es un vídeo roto.
        adjusted = ((arguments.frames - 1) // 8) * 8 + 9
        print(
            f"AVISO: LTX-Video espera fotogramas de la forma 8k+1 (9, 17, 25, 49, 97); "
            f"{arguments.frames} se ajusta a {adjusted}.",
            file=sys.stderr,
        )
        arguments.frames = adjusted

    try:
        import torch
        from diffusers import LTXPipeline
    except ImportError as error:
        return fail(
            f"Falta {error.name}. Instala el extra de vídeo con: "
            "uv pip install torch diffusers transformers accelerate imageio imageio-ffmpeg"
        )

    # Before loading anything: the version failure otherwise surfaces 80 seconds
    # later, once the tokenizer has already pulled half a gigabyte of weights.
    incompatible = _check_transformers_compatibility()
    if incompatible:
        return fail(incompatible)

    if not torch.cuda.is_available():
        return fail("No hay CUDA disponible. LTX-Video en CPU es impracticable.")

    output = _output_path(arguments.salida, arguments.nombre, "mp4")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    started = time.time()

    try:
        pipeline = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    except Exception as error:  # noqa: BLE001
        return fail(f"No se pudieron cargar los pesos de {MODEL_ID}: {error}")

    # No se llama a `pipeline.to("cuda")` antes del offload, y esto no es una
    # omisión: los tres modos de offload gestionan la residencia de las capas y
    # necesitan que el modelo esté en CPU para bajarlo. Con `to("cuda")` delante
    # el pipeline entero (unos 7 GB) queda en la tarjeta y el offload no tiene
    # de dónde bajar nada: es un OOM antes de empezar a renderizar.
    #
    # Tampoco se sube el VAE a fp32. Ayuda un poco de calidad en clips largos,
    # pero se come VRAM, que aquí es el recurso escaso.
    apply_offload(pipeline, arguments.offload)
    # En diffusers 0.39 el VAE de LTX no tiene tiling/slicing; la palanca que
    # sí existe para el pico de memoria de la atención es esta.
    pipeline.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(arguments.semilla)
    try:
        frames = pipeline(
            prompt=arguments.prompt,
            negative_prompt=arguments.negativo,
            num_frames=arguments.frames,
            num_inference_steps=arguments.steps,
            width=arguments.width,
            height=arguments.height,
            generator=generator,
        ).frames[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fail(
            "Se acabó la VRAM durante el render. Baja --frames (17 o 25), o usa "
            "--offload sequential. Los pasos no afectan a la memoria."
        )

    save_video(frames, output, arguments.fps)
    elapsed = time.time() - started

    # La portada en PNG es lo que el usuario mira primero, y la guarda el
    # mismo render: no hay que volver a generar para tener una miniatura.
    poster: Path | None = output.with_suffix(".png")
    try:
        frames[0].save(poster)
    except Exception:  # noqa: BLE001 - el png es una comodidad, no el resultado
        poster = None

    emit(
        {
            "ok": True,
            "path": str(output),
            "poster": str(poster) if poster else "",
            "frames": arguments.frames,
            "steps": arguments.steps,
            "resolution": f"{arguments.width}x{arguments.height}",
            "offload": arguments.offload,
            "elapsed": round(elapsed, 1),
            "vram_estimate_mib": estimate_vram_mib(arguments.frames, arguments.offload),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

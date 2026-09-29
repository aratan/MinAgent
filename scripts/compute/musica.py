#!/usr/bin/env python3
"""Genera música o ambiente con MusicGen, y cabe en 8 GB de VRAM.

MusicGen es el modelo de audio de Meta que hace los tres tamaños: small (300M),
medium (1.5B) y large (3.3B). Aquí se usa small por una razón concreta, no por
costumbre: medium y large necesitan entre 6 y 9 GB solo en los pesos, así que
en una tarjeta de 8 GB que además tiene los motores de voz residentes, no caben
aunque no haya nada más corriendo. Small entra con holgura y genera
audio de 32 kHz que sirve para música de fondo, intros y ambiente.

Es la diferencia real entre los tres: la calidad sube, el VRAM también, y el
límite de esta tarjeta está en 8 GB, no en lo que se pueda pedir.

Dos detalles que no son obvios:

1. La duración la fija el número de tokens de audio, y el audio se genera en
   bloques de 50 tokens. Pedir 30 segundos son 1500 tokens, y eso son 1500 pasos
   de decodificación sobre el mismo contexto: más tiempo, no más VRAM. Lo que
   sí crece con la duración es el contexto de conditioning, y despacio.
2. `guidance_scale` es lo que separa "música" de "ruido con forma". El valor por
   defecto de MusicGen (3.5) está bien para música; para ambiente y sonido de
   fondo, bajarlo a 2 da algo más difuso y menos competente, que es justo lo
   que se quiere de un fondo.

La salida es un wav en `salida/`, a 32 kHz, listo para usar.

Ejemplos:
    python scripts/compute/musica.py --prompt "lo-fi hip hop, warmRhodes, relaxed"
    python scripts/compute/musica.py --prompt "epic orchestral battle drums" --segundos 20
    python scripts/compute/musica.py --prompt "..." --modelo medium
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Debe fijarse antes de que torch inicialice CUDA: si torch ya se ha importado,
# CUDA ha reservado su pool con la configuración por defecto y esto llega tarde.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_IDS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
}

SAMPLE_RATE = 32_000
TOKENS_PER_SECOND = 50
"""Un token de audio son 20 ms: 50 tokens es un segundo de sonido."""

# Cuánta VRAM hace falta por tamaño, ya descontando el offload. `small` es el
# único que entra con los motores de voz residentes; los otros dos son la razón
# por la que el orquestador los rechaza cuando la tarjeta está ocupada.
VRAM_ESTIMATE_MIB = {
    "small": 2800,
    "medium": 6500,
    "large": 9500,
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "salida"


def emit(payload: dict) -> None:
    print("RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)


def fail(message: str, code: int = 1) -> int:
    emit({"ok": False, "error": message})
    return code


def _output_path(requested: str, stem: str) -> Path:
    """Resolve the wav path inside salida/, the only place generated audio lands."""
    name = (requested or "").strip()
    if not name:
        name = f"{stem}-{int(time.time())}.wav"
    elif not Path(name).suffix:
        name = f"{name}.wav"
    path = Path(name)
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera música con MusicGen.")
    parser.add_argument("--prompt", required=True, help="Qué debe sonar.")
    parser.add_argument("--segundos", type=int, default=15, help="Duración aproximada en segundos.")
    parser.add_argument(
        "--modelo",
        default="small",
        choices=sorted(MODEL_IDS),
        help=(
            "'small' es el único que cabe en 8 GB junto a los motores de voz; "
            "'medium' y 'large' necesitan una tarjeta con más VRAM libre."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Guidance scale. 3.5 para música; 2 da ambiente más difuso.",
    )
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", default="", help="Ruta del wav de salida.")
    parser.add_argument("--nombre", default="musica", help="Nombre base de la salida.")
    arguments = parser.parse_args(argv)

    if arguments.segundos < 1:
        return fail("La duración mínima es 1 segundo.")
    if arguments.segundos > 120:
        return fail("La duración máxima por trabajo es 120 segundos; encadena varios si hace falta.")

    try:
        import torch
        from diffusers import MusicgenPipeline
    except ImportError as error:
        return fail(
            f"Falta {error.name}. Instala el extra de música con: "
            "uv pip install torch diffusers transformers accelerate scipy"
        )

    if not torch.cuda.is_available():
        return fail("No hay CUDA disponible. MusicGen en CPU tarda horas.")

    tokens = arguments.segundos * TOKENS_PER_SECOND
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    output = _output_path(arguments.salida, arguments.nombre)
    started = time.time()

    try:
        pipeline = MusicgenPipeline.from_pretrained(
            MODEL_IDS[arguments.modelo], torch_dtype=dtype, guidance_scale=arguments.guidance
        )
    except Exception as error:  # noqa: BLE001
        return fail(f"No se pudieron cargar los pesos de {MODEL_IDS[arguments.modelo]}: {error}")

    # Offload de modelo y no secuencial: los pesos de MusicGen son un único
    # bloque, así que no hay submodelos que ir moviendo por grupos. Lo que se
    # mueve a RAM son las activaciones durante la decodificación, que es donde
    # una duración larga se nota.
    pipeline.enable_model_cpu_offload()  # type: ignore[attr-defined]

    generator = torch.Generator(device="cpu").manual_seed(arguments.semilla)
    try:
        audio = pipeline(
            arguments.prompt,
            num_tokens=tokens,
            num_inference_steps=tokens,
            do_sample=True,
            guidance_scale=arguments.guidance,
            generator=generator,
        ).audios[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fail(
            "Se acabó la VRAM generando el audio. Prueba con --modelo small y menos segundos, "
            "o libera la memoria que tengan otros procesos."
        )

    import numpy as np
    import soundfile as sf

    samples = audio.squeeze().cpu().numpy().astype(np.float32)
    sf.write(str(output), samples, SAMPLE_RATE)

    emit(
        {
            "ok": True,
            "path": str(output),
            "model": arguments.modelo,
            "seconds": round(len(samples) / SAMPLE_RATE, 2),
            "elapsed": round(time.time() - started, 1),
            "vram_estimate_mib": VRAM_ESTIMATE_MIB[arguments.modelo],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

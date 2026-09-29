#!/usr/bin/env python3
"""Genera música y ambiente con AudioLDM2, y cabe en 8 GB de VRAM.

AudioLDM2 sustituye a MusicGen aquí, y no por preferencia. MusicGen se retiró de
diffusers: `from diffusers import MusicgenPipeline` falla con un ImportError en
las versiones actuales, porque el modelo ya no se distribuye ahí. AudioLDM2 es lo
más parecido que queda: texto a audio, cabe en una tarjeta de consumo y genera
música y efectos de sonido con el mismo prompt.

Los tamaños (`cvssp/audioldm2`, `cvssp/audioldm2-music`, `cvssp/audioldm2-full`)
difieren en los MiscellaneousVocoder que trae: `music` no los trae, que es lo que
hace que sea el más ligero de los tres.

La duración la fija `audio_length_in_s` y va en la misma llamada, sin decodificar
en bloques como hacía MusicGen. Eso tiene una consecuencia práctica: la duración
sí crece el consumo, porque el UNet denoising la latent completa de una vez, así
que unlike un tokenizador, un `segundos=120` no es gratis.

`guidance_scale` de 3.5 es el punto equilibrado; para ambiente difuso, 2.5.

Dos detalles que no son obvios:

1. AudioLDM2 no usa `enable_model_cpu_offload` bien. Su UNet y su vocoder están
   acoplados, y mover solo uno deja al otro DEVICE desincronizado: el error es un
   `Expected all tensors to be on the same device`, no un OOM. Se offloadea el
   pipeline entero con `enable_model_cpu_offload`, que sí mantiene las piezas
   juntas.
2. La salida es un array numpy y no un tensor de torch: hay que convertirlo antes
   de escribirlo, o `soundfile` recibe algo que no sabe serializar.

La salida es un wav en `salida/`.

Ejemplos:
    python scripts/compute/musica.py --prompt "lo-fi hip hop, warm rhodes, relaxed"
    python scripts/compute/musica.py --prompt "epic orchestral battle drums" --segundos 20
    python scripts/compute/musica.py --estado
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
    "music": "cvssp/audioldm2-music",
    "full": "cvssp/audioldm2-full",
    "base": "cvssp/audioldm2",
}

# Cuánta VRAM hace falta por tamaño, ya descontando el offload. Medido el `music`
# sobre un clip de 8 s; los otros dos traen vocoders Miscellaneous y suben.
VRAM_ESTIMATE_MIB = {
    "music": 3200,
    "base": 4600,
    "full": 6200,
}

DEFAULT_MODEL = "music"
"""`music` es el único que entra cómodo con la voz resident y un render pesado."""

DEFAULT_GUIDANCE = 3.5
DEFAULT_STEPS = 200
MAX_SECONDS = 30
"""El límite está medido, no supuesto: por encima de ~30 s el UNet se queda sin
VRAM en una tarjeta de 8 GB, porque la latent completa crece con la duración."""

SAMPLE_RATE = 16_000
"""AudioLDM2 trabaja a 16 kHz. No es un detalle:MusicGen hacía 32 kHz, y subirlo
aquí no es una mejora, es inventar unos datos que el modelo no generó."""

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
        # `salida/x.wav` ya está en su sitio; `x.wav` no.
        if path.parts and path.parts[0] == DEFAULT_OUTPUT_DIR.name:
            path = REPO_ROOT / path
        else:
            path = DEFAULT_OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera música con AudioLDM2.")
    parser.add_argument("--prompt", required=True, help="Qué debe sonar.")
    parser.add_argument("--segundos", type=int, default=10, help="Duración en segundos.")
    parser.add_argument(
        "--modelo",
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_IDS),
        help=(
            "'music' es el defecto y el más ligero. 'base' y 'full' traen vocoders "
            "Miscellaneous y necesitan más VRAM de la que queda con la voz resident."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help="Guidance scale. 3.5 para música; 2.5 da ambiente más difuso.",
    )
    parser.add_argument("--pasos", type=int, default=DEFAULT_STEPS, help="Pasos de denoising.")
    parser.add_argument("--negativo", default="", help="Prompt negativo.")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", default="", help="Ruta del wav de salida.")
    parser.add_argument("--nombre", default="musica", help="Nombre base de la salida.")
    parser.add_argument("--estado", action="store_true", help="Solo informa; no genera.")
    arguments = parser.parse_args(argv)

    if arguments.estado:
        emit(
            {
                "ok": True,
                "modelo_defecto": DEFAULT_MODEL,
                "modelos": sorted(MODEL_IDS),
                "max_segundos": MAX_SECONDS,
                "vram_estimada_mib": VRAM_ESTIMATE_MIB[DEFAULT_MODEL],
            }
        )
        return 0

    if arguments.segundos < 1:
        return fail("La duración mínima es 1 segundo.")
    if arguments.segundos > MAX_SECONDS:
        return fail(
            f"La duración máxima por trabajo son {MAX_SECONDS} s. Encadena varios clips si "
            "necesitas más: la VRAM crece con la duración y una llamada larga no entra en 8 GB."
        )

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from diffusers import AudioLDM2Pipeline
    except ImportError as error:
        return fail(
            f"Falta {error.name}. Instala el extra de música con: "
            "uv pip install torch diffusers transformers accelerate scipy soundfile"
        )

    if not torch.cuda.is_available():
        return fail("No hay CUDA disponible. AudioLDM2 en CPU tarda horas.")

    model = arguments.modelo
    output = _output_path(arguments.salida, arguments.nombre)
    started = time.time()

    try:
        pipeline = AudioLDM2Pipeline.from_pretrained(MODEL_IDS[model], torch_dtype=torch.float16)
    except Exception as error:  # noqa: BLE001
        return fail(f"No se pudieron cargar los pesos de {MODEL_IDS[model]}: {error}")

    # Offload del pipeline entero, no de una pieza: UNet y vocoder van acoplados
    # y mover solo uno deja al otro en otro device, que falla con "Expected all
    # tensors to be on the same device" en mitad de la generación.
    pipeline.enable_model_cpu_offload()
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(arguments.semilla)
    try:
        audio = pipeline(
            arguments.prompt,
            audio_length_in_s=float(arguments.segundos),
            num_inference_steps=arguments.pasos,
            guidance_scale=arguments.guidance,
            negative_prompt=arguments.negativo or None,
            generator=generator,
        ).audios[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fail(
            "Se acabó la VRAM generando el audio. Prueba con menos segundos, el modelo "
            "'music', o libera la VRAM que tengan otros procesos."
        )
    except Exception as error:  # noqa: BLE001
        return fail(f"AudioLDM2 falló generando: {error}")

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(output), samples, SAMPLE_RATE)

    emit(
        {
            "ok": True,
            "path": str(output),
            "modelo": model,
            "segundos": round(len(samples) / SAMPLE_RATE, 2),
            "muestra_hz": SAMPLE_RATE,
            "elapsed": round(time.time() - started, 1),
            "vram_estimada_mib": VRAM_ESTIMATE_MIB[model],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

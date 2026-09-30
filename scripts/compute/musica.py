#!/usr/bin/env python3
"""Genera música y ambiente con MusicGen, y cabe en 8 GB de VRAM.

Usa `facebook/musicgen-small` por la vía nativa de transformers, no por diffusers.
No es arbitrario: es el único camino que funciona hoy, y se comprobó de las dos
formas.

diffusers retiró `MusicgenPipeline` y `MusicgenMelodyPipeline`; en la versión
instalada (0.40.0) esos nombres no existen. Y AudioLDM2, el sustituto que había
en su lugar, tampoco aguanta: su pipeline llama a
`_update_model_kwargs_for_generation`, una API privada de transformers que
cambió en transformers 5.x, así que revienta con

    'GPT2Model' object has no attribute '_update_model_kwargs_for_generation'

Es un fallo de diffusers contra transformers 5, no del modelo ni de la tarjeta.
Como el modelo de Meta se carga directamente con transformers, no depende de esa
API y sí funciona: 4.94 s generados en 3.2 s de GPU con 1303 MiB de pico.

Detalles que no son obvios:

1. La duración se pide en TOKENS, no en segundos. MusicGen decodifica a 32 kHz
   con 50 tokens por segundo, así que 5 s son `max_new_tokens=250`. Pasarle
   segundos directamente genera un clip de la duración equivocada sin avisar.
2. `sample_rate` y `hop_length` se leen de la config del modelo, no se fijan a
   mano. AudioLDM2 trabajaba a 16 kHz y clavar 32000 aquí sería inventar datos que
   el modelo no generó. La frecuencia real se lee del audio que sale.
3. No hay prompt negativo. MusicGen no lo soporta: `--negativo` se retiró del
   contrato en vez de aceptarse y no hacer nada, que es peor que no tenerlo.
4. No hay pasos de denoising: no es un UNet. El único dial que de verdad cambia
   el resultado es `guidance_scale`.

Un fallo silencioso que sí se comprueba: un modelo puede devolver audio digital
silente sin dar error. Se mide la señal de la salida y un resultado mudo se
reporta como fallo, porque un `ok` con silencio es peor que un error.

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

import numpy as np

# Debe fijarse antes de que torch inicialice CUDA: si torch ya se ha importado,
# CUDA ha reservado su pool con la configuración por defecto y esto llega tarde.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_IDS = {
    "small": "facebook/musicgen-small",
}
"""Un solo modelo, y es a propósito.

`musicgen-medium` (1.5B) cabría en 8 GB, pero no se ha medido aquí, y publicar una
estimación sin medir es publicar una capacidad que luego revienta a mitad de clip.
Un modelo probado vale más que dos, donde uno sea una suposición.
"""

DEFAULT_MODEL = "small"

VRAM_ESTIMATE_MIB = {
    "small": 2200,
}
"""VRAM que necesita el job de música, en MiB.

Medido: 1303 MiB de pico en un clip de 5 s con fp16. Se redondea a la baja
round-up con margen porque la estimación va antes de la carga: si se queda
corta, el fallo aparece a mitad de generación, que es donde más cuesta.
La duración no entra aquí de forma lineal: la caché de atención crece con los
tokens, pero un clip de 30 s no usa 6x lo de uno de 5 s.
"""

DEFAULT_GUIDANCE = 3.0
MAX_SECONDS = 30
"""Techo razonado, no medido a 30 s: 5 s usaron 1303 MiB y 30 s son 1500 tokens
frente a 250, o sea 6x la secuencia. La atención es cuadrática en la longitud,
pero la caché de 1500 tokens sigue siendo pequeña para 8 GB. Si alguna vez se
supera, el propio OOM cae aquí y lo dice."""

TOKENS_PER_SECOND = 50
"""A 32 kHz, MusicGen decodifica 50 tokens por segundo. Se usa como respaldo si
la config del modelo no expone `hop_length`; lo normal es leerlo de ahí."""

SILENCE_RMS = 1e-4
"""Por debajo de este RMS la salida es silencio digital, no música. El umbral es
deliberadamente bajo: hasta una pieza muy suave tiene energía de sobra."""

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


def _token_count(model_config, seconds: int) -> int:
    """Tokens que hay que pedir, leídos de la config del modelo y no supuestos.

    MusicGen es autoregresivo sobre tokens de audio, así que la longitud de lo que
    se pide es la longitud del clip. equivocarse aquí da un clip de otra duración
    sin ningún error, que es justo el fallo que sobrevive a un test que solo
    comprueba `ok: true`.
    """
    encoder = getattr(model_config, "audio_encoder", None)
    sample_rate = getattr(encoder, "sampling_rate", 0) or 0
    hop = getattr(encoder, "hop_length", 0) or 0
    if sample_rate and hop:
        return max(1, round(sample_rate / hop * seconds))
    return TOKENS_PER_SECOND * seconds


def _to_samples(audio) -> np.ndarray:
    """Convierte lo que devuelve `generate()` en algo que soundfile sepa escribir.

    Los tres pasos no son decorativos y el orden importa: `detach` saca el grafo
    de autograd, `float` baja de fp16 a fp32 porque numpy no tiene float16, y
    `cpu` mueve el tensor de la GPU a memoria del host. Sin cualquiera de ellos
    el job revienta: con `cpu` de menos, "can't convert cuda:0 device type tensor
    to numpy"; con `float` de menos, "can't convert Half to numpy".
    """
    return audio[0, 0].detach().float().cpu().numpy().reshape(-1)


def _rms(samples: np.ndarray) -> float:
    """Root mean square de la señal, en un array de float ya en el host."""
    return float(np.sqrt((samples.astype(np.float64) ** 2).mean()))


def _is_silent(samples: np.ndarray) -> bool:
    """True cuando la salida es silencio digital y no música.

    Un modelo puede devolver audio mudo sin lanzar ningún error, así que la señal
    se mide. Un `ok` con un wav en silencio es indistinguible de un acierto, que
    es la razón por la que esto se comprueba en vez de confiar.
    """
    return bool(_rms(samples) < SILENCE_RMS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera música con MusicGen.")
    parser.add_argument(
        "--prompt", default="", help="Qué debe sonar. Opcional solo con --estado."
    )
    parser.add_argument("--segundos", type=int, default=10, help="Duración en segundos.")
    parser.add_argument(
        "--modelo",
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_IDS),
        help="'small' es el único medido y el único disponible.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help="Guidance scale. 3.0 es el valor de MusicGen; subirlo fuerza más al prompt.",
    )
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

    if not arguments.prompt.strip():
        return fail("Falta --prompt: dice qué debe sonar.")

    if arguments.segundos < 1:
        return fail("La duración mínima es 1 segundo.")
    if arguments.segundos > MAX_SECONDS:
        return fail(
            f"La duración máxima por trabajo son {MAX_SECONDS} s. Encadena varios clips si "
            "necesitas más: la secuencia crece con la duración y un clip largo no entra cómodo."
        )

    try:
        import soundfile as sf
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError as error:
        return fail(
            f"Falta {error.name}. Instala el extra de música con: "
            "uv pip install torch transformers accelerate soundfile numpy"
        )
    except (OSError, RuntimeError) as error:
        # No es un paquete que falte: es una biblioteca nativa que no enlaza.
        # La causa real -y la que costó una sesión entera de diagnóstico- es que
        # torchaudio venga con un CUDA distinto al de torch: la extensión
        # `_torchaudio.abi3.so` se compila contra `libcudart.so.12` o `.13`
        # según la wheel, y si no coincide con la de torch el import falla con
        # "Could not load this library", no con "No module named". Reinstalar la
        # lista de paquetes no arregla nada; hay que alinear las dos wheels.
        detalle = str(error).strip().splitlines()[-1] if str(error).strip() else repr(error)
        # `torch` puede no estar ligado: si el que falló fue su propio import,
        # leer la versión aquí volvería a fallar y taparía el motivo real.
        torch_mod = sys.modules.get("torch")
        version = getattr(torch_mod, "__version__", None) or ""
        cuda = getattr(torch_mod, "version", None)
        cuda = getattr(cuda, "cuda", None) if cuda else None
        if not (version and cuda):
            return fail(
                "La biblioteca nativa de audio no carga, así que esto no es un problema del "
                f"modelo. Causa: {detalle}. No se puede deducir aquí la build de torch, así que "
                "alinea torchaudio con la variante de CUDA que usa tu torch, mirando su versión "
                "local (por ejemplo 2.11.0+cu128 se reinstala desde el índice cu128)."
            )
        tag = "cu" + cuda.replace(".", "")
        return fail(
            "La biblioteca nativa de audio no carga, así que esto no es un problema del modelo. "
            f"Causa: {detalle}. Suele ser torch y torchaudio compilados contra versiones distintas "
            "de CUDA; se arregla reinstalando torchaudio desde el índice que corresponde al torch "
            f"de este venv (torch aquí es {version}):\n"
            "    uv pip install --reinstall --no-deps torchaudio"
            f"=={version.split('+')[0]}+{tag} --index-url https://download.pytorch.org/whl/{tag}"
        )

    if not torch.cuda.is_available():
        return fail("No hay CUDA disponible. MusicGen en CPU es impracticable.")

    model_name = MODEL_IDS[arguments.modelo]
    output = _output_path(arguments.salida, arguments.nombre)
    started = time.time()

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to("cuda")
    except Exception as error:  # noqa: BLE001
        return fail(f"No se pudieron cargar los pesos de {model_name}: {error}")

    sample_rate = int(getattr(model.config.audio_encoder, "sampling_rate", 0) or 32_000)
    max_new_tokens = _token_count(model.config, arguments.segundos)
    torch.manual_seed(arguments.semilla)

    try:
        inputs = processor(
            text=[arguments.prompt], padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            audio = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=arguments.guidance,
                max_new_tokens=max_new_tokens,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return fail(
            "Se acabó la VRAM generando el audio. Prueba con menos segundos o libera la VRAM "
            "que tengan otros procesos."
        )
    except Exception as error:  # noqa: BLE001
        return fail(f"MusicGen falló generando: {error}")

    # detach/float/cpu: mira `_to_samples`, los tres pasos son obligatorios.
    samples = _to_samples(audio)
    rms = _rms(samples)
    if _is_silent(samples):
        return fail(
            f"El modelo devolvió silencio (rms {rms:.2e}), no música. Se reporta como fallo y no "
            "como un ok con un wav vacío, porque un ok mudo no lo distingue nadie de un acierto."
        )

    sf.write(str(output), samples, sample_rate)

    emit(
        {
            "ok": True,
            "path": str(output),
            "modelo": arguments.modelo,
            "segundos": round(len(samples) / sample_rate, 2),
            "muestra_hz": sample_rate,
            "tokens": max_new_tokens,
            "rms": round(rms, 5),
            "pico": round(float(np.abs(samples).max()), 4),
            "elapsed": round(time.time() - started, 1),
            "vram_pico_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "vram_estimada_mib": VRAM_ESTIMATE_MIB[arguments.modelo],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

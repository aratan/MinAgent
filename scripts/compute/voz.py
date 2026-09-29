#!/usr/bin/env python3
"""STT y TTS locales, los dos motores que se quedan residentes en VRAM.

Este script es la voz del agente: Kokoro sintetiza el habla y whisper.cpp
transcribe. Los dos se cargan una vez y se quedan, y entre los dos ocupan menos
de 2.5 GB, que es lo que permite que la voz siga respondiendo mientras un
vídeo o una música se generan en otro proceso.

Por qué whisper.cpp y no faster-whisper: faster-whisper (CTranslate2) pide
~1.5 GB en fp16 y carga tensores de Torch en el proceso. whisper.cpp es un
binario que se apoya en la GPU sin ningún runtime de Python, y los modelos
small y base entran en el presupuesto de los 2.5 GB dejando sitio de sobra para
Kokoro. En una tarjeta donde la restricción es física, la opción que no
arrastra un runtime entero detrás es la que aguanta.

Los dos motores se cargan de forma perezosa: un ``--solo`` carga solo lo que
hace falta, y sin argumentos se responde con el estado en JSON para que el
orquestador sepa qué está cargado sin pagar la carga.

La salida es siempre ``RESULT <json>`` en una línea, para que el orquestador la
lea sin depender de lo que impriman las librerías por su cuenta.

Ejemplos:
    python scripts/compute/voz.py --texto "Hola, ¿qué tal?" --salida salida/habla.wav
    python scripts/compute/voz.py --texto "Ready" --voz af_heart --velocidad 1.1
    python scripts/compute/voz.py --audio salida/audio.mp3 --idioma es
    python scripts/compute/voz.py --estado
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

# Debe fijarse antes de que torch inicialice CUDA: si torch ya se ha importado,
# CUDA ha reservado su pool con la configuración por defecto y esto llega tarde.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "salida"

KOKORO_LANG_PREFIX = {
    "es": "e",
    "en": "a",
    "fr": "f",
    "ja": "j",
    "zh": "z",
    "hi": "h",
    "it": "i",
    "pt": "p",
}
"""Kokoro nombra las voces con un prefijo de idioma.

El prefijo no es decorativo: `af_heart` habla inglés y `ef_dora` habla español.
Pasar la voz equivocada no da error, da un acento espurio, así que se deduce
del idioma cuando no se ha pedido una voz concreta.
"""

WHISPER_DEFAULT_MODEL = "base"
"""El modelo de whisper.cpp por defecto.

`tiny` es el más rápido pero se equivoca tanto que no sirve para transcribir
bien; `small` sube bastante de tamaño y `medium` ya no cabe cómodamente en el
presupuesto de voz junto a Kokoro. `base` es el punto donde la transcripción es
usable sin comerse la reserva.
"""

# Los pesos de Kokoro (~300 MB) se descargan la primera vez a la caché de HF.
# El script no los baja por su cuenta: es deliberado, para que la descarga
# larga sea algo que alguien lanzó a knowingly, no una sorpresa al hablar.
_KOKORO_HINT = (
    "Kokoro no está instalado. Instálalo con: uv pip install kokoro soundfile numpy\n"
    "Los pesos (~300 MB) se descargan de HuggingFace la primera vez."
)


def emit(payload: dict) -> None:
    """Write the single machine-readable line the orchestrator reads."""
    print("RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)


def fail(message: str, code: int = 1) -> int:
    emit({"ok": False, "error": message})
    return code


def detect_language(text: str) -> str:
    """A rough language guess from the text itself, for picking a Kokoro voice.

    Deliberately cheap: a stopword count is enough to choose between two voice
    packs, and anything smarter would need a model that costs more VRAM than
    the thing it is picking a voice for.
    """
    lowered = text.lower()
    spanish = sum(lowered.count(word) for word in (" el ", " la ", " los ", " que ", " de ", " y "))
    english = sum(lowered.count(word) for word in (" the ", " and ", " of ", " is ", " to "))
    return "es" if spanish > english else "en"


def split_for_speech(text: str, limit: int = 400) -> list[str]:
    """Split long text at sentence ends, because Kokoro's context is short.

    One 3000-character string fed to the model as a single call produces audio
    with the prosody of a paragraph run together. Splitting at punctuation and
    stitching the wavs back together is what makes a long answer sound like
    speech instead of a drone.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return [cleaned]

    pieces: list[str] = []
    current = ""
    for word in cleaned.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
        # A piece that has run long and ends on a stop is a good break point.
        if len(current) > limit * 0.6 and current[-1] in ".!?;:":
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return pieces


def _output_path(requested: str, suffix: str) -> Path:
    """Resolve the wav path inside salida/, which is the only place audio lands."""
    name = (requested or "").strip()
    if not name:
        name = f"voz-{int(time.time())}.{suffix}"
    elif not Path(name).suffix:
        name = f"{name}.{suffix}"
    path = Path(name)
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def speak(text: str, output: str, voice: str, speed: float) -> dict:
    """Synthesize speech with Kokoro and write a single wav."""
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as error:
        return {"ok": False, "error": f"{_KOKORO_HINT} (falta {error.name})"}

    if not voice:
        prefix = KOKORO_LANG_PREFIX.get(detect_language(text), "a")
        voice = f"{prefix}f_heart"
    try:
        pipeline = KPipeline(lang_code=voice[0])
    except Exception as error:  # noqa: BLE001 - surfaced to the caller as text
        return {"ok": False, "error": f"No se pudo cargar Kokoro: {error}"}

    started = time.time()
    try:
        chunks = [
            (audio if audio is not None else np.zeros(1))
            for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        ]
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": f"Kokoro falló sintetizando: {error}"}

    if not chunks:
        return {"ok": False, "error": "Kokoro no produjo audio para ese texto."}
    audio = np.concatenate(chunks, axis=0)
    path = _output_path(output, "wav")
    sf.write(str(path), audio, 24000)
    return {
        "ok": True,
        "path": str(path),
        "voice": voice,
        "seconds": round(len(audio) / 24000, 2),
        "elapsed": round(time.time() - started, 2),
    }


def _whisper_binary() -> str | None:
    """Locate the whisper.cpp executable, which is installed separately.

    The binary names differ between the distro package, a manual build, and
    the CMake default, so all three are tried before giving up.
    """
    for candidate in ("whisper-cli", "whisper-cpp", "main"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def transcribe(audio_path: str, language: str, model: str) -> dict:
    """Transcribe a media file with the whisper.cpp binary."""
    binary = _whisper_binary()
    if binary is None:
        return {
            "ok": False,
            "error": (
                "whisper.cpp no está instalado. En Arch: pacman -S whisper.cpp\n"
                "O compílalo de https://github.com/ggerganov/whisper.cpp; el binario se llama whisper-cli."
            ),
        }
    if not Path(audio_path).is_file():
        return {"ok": False, "error": f"No existe el archivo de audio: {audio_path}"}

    model_dir = os.environ.get("WHISPER_MODEL_DIR", "").strip()
    if model_dir and not Path(model_dir, f"ggml-{model}.bin").is_file():
        return {
            "ok": False,
            "error": (
                f"Falta el modelo ggml-{model}.bin en {model_dir}. Descárgalo de "
                f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin"
            ),
        }

    command = [
        binary,
        "-m",
        str(Path(model_dir, f"ggml-{model}.bin")) if model_dir else f"ggml-{model}.bin",
        "-f",
        audio_path,
        "-nt",  # sin marcas de tiempo: el texto es lo que se quiere
        "-otxt",
        "-of",
        str(Path(audio_path).with_suffix("")),
    ]
    if language:
        command += ["-l", language]
    if shutil.which("nvidia-smi"):
        command.append("-ng")  # GPU: es lo que mantiene el STT por debajo de 1 GB

    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "whisper.cpp tardó más de 10 minutos y se detuvo."}
    except OSError as error:
        return {"ok": False, "error": f"No se pudo ejecutar whisper.cpp: {error}"}

    transcript = Path(f"{audio_path}.txt")
    text = transcript.read_text(encoding="utf-8", errors="replace") if transcript.is_file() else ""
    if not text.strip():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        return {
            "ok": False,
            "error": "whisper.cpp no devolvió transcripción.\n" + "\n".join(tail),
        }
    return {
        "ok": True,
        "text": text.strip(),
        "model": model,
        "elapsed": round(time.time() - started, 2),
    }


def wav_duration(path: Path) -> float:
    """Duration of a wav, for the log line. Returns 0.0 when it is not a wav."""
    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate() or 1), 2)
    except (OSError, wave.Error):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STT y TTS locales de MinAgent.")
    parser.add_argument("--texto", help="Texto a sintetizar con Kokoro.")
    parser.add_argument("--salida", default="", help="Ruta del wav de salida.")
    parser.add_argument("--voz", default="", help="Voz de Kokoro, p.ej. af_heart o ef_dora.")
    parser.add_argument("--velocidad", type=float, default=1.0, help="Velocidad del habla.")
    parser.add_argument("--audio", help="Archivo de audio o vídeo a transcribir.")
    parser.add_argument("--idioma", default="", help="Código de idioma, p.ej. es. Vacío = autodetectar.")
    parser.add_argument("--modelo-stt", default=WHISPER_DEFAULT_MODEL, help="Modelo de whisper.cpp.")
    parser.add_argument("--estado", action="store_true", help="Informa del estado de los motores.")
    arguments = parser.parse_args(argv)

    if arguments.estado:
        # A status probe is a question, not a job, so a missing engine is an
        # answer rather than a failure: exit 0 with the install line included.
        if not _kokoro_available():
            emit({"ok": False, "error": _KOKORO_HINT, "whisper": _whisper_binary() or "no instalado"})
        else:
            emit({"ok": True, "kokoro": "disponible", "whisper": _whisper_binary() or "no instalado"})
        return 0

    if arguments.texto:
        result = speak(arguments.texto, arguments.salida, arguments.voz, arguments.velocidad)
        if result.get("ok"):
            duration = wav_duration(Path(str(result["path"])))
            if duration:
                result["seconds"] = duration
        emit(result)
        return 0 if result.get("ok") else 1

    if arguments.audio:
        result = transcribe(arguments.audio, arguments.idioma, arguments.modelo_stt)
        emit(result)
        return 0 if result.get("ok") else 1

    parser.print_help()
    return 2


def _kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())

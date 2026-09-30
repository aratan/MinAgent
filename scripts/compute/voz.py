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
import re
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

KOKORO_VOICES = {
    "es": "ef_dora",
    "en": "af_heart",
    "fr": "ff_siwis",
    "ja": "jf_alpha",
    "zh": "zf_xiaobei",
    "hi": "hf_alpha",
    "it": "if_sara",
    "pt": "pf_dora",
}
"""Voz por defecto de cada idioma, tal y como existen en hexgrad/Kokoro-82M.

Estos nombres están tomados del repositorio, no inventados. Importa: una voz
que no existe no da un error claro, da un 404 de HuggingFace en mitad de la
síntesis, y `ef_heart` -que fue el primer valor probado- no existe: en español
solo hay `ef_dora`. Para inglés hay muchas (`af_bella`, `af_nicole`,
`af_sarah`...) y se elige `af_heart` por ser la más neutra.

El prefijo no es decorativo: `af_heart` habla inglés y `ef_dora` español. Pasar
la voz equivocada no da error, da un acento espurio, así que se deduce del
idioma cuando no se ha pedido una concreta.
"""

WHISPER_DEFAULT_MODEL = "base"
"""El modelo de whisper.cpp por defecto.

`tiny` es el más rápido pero se equivoca tanto que no sirve para transcribir
bien; `small` sube bastante de tamaño y `medium` ya no cabe cómodamente en el
presupuesto de voz junto a Kokoro. `base` es el punto donde la transcripción es
usable sin comerse la reserva.
"""

EN_SPACY_MODEL = "en_core_web_sm"
"""El modelo de spaCy que misaki usa para el inglés.

Kokoro no trae los datos de spaCy: el paquete `en_core_web_sm` se instala
aparte. Sin él, `KPipeline(lang_code="a")` falla al construirse con un
"Can't find model", que no dice que lo que falta es un modelo de lenguaje ni
cómo instalarlo. Se instala aquí la primera vez, y solo si se va a hablar en
inglés: en español no hace falta.
"""

EN_SPACY_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
"""La rueda directa, no `python -m spacy download`.

`spacy download` imprimió "Download and installation successful" sin instalar
nada en este equipo, y el fallo solo apareció al intentar cargar el modelo
después. Instalar la URL directamente sí funciona, y el mensaje que ve quien
lee el error dice exactamente eso.
"""
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
    'Kokoro no está instalado. Instálalo con la versión fijada: '
    'uv pip install "kokoro==0.9.4" soundfile numpy\n'
    "Sin esa versión no funciona: 0.7.x quitó `repo_id` a KPipeline y devuelve un "
    "clip fijo de 0,25 s. Los pesos (~300 MB) se descargan de HuggingFace la primera vez."
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

    The words are matched as whole words rather than as substrings. That
    matters: "esta" contains "es" and "de" is a stopword in both languages, so
    a substring count calls "Hola, esto es una prueba" English and hands it to
    an English voice. Getting a Spanish sentence read with an English accent is
    worse than the rough estimate this is.
    """
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    spanish = sum(words.count(word) for word in SPANISH_MARKERS)
    english = sum(words.count(word) for word in ENGLISH_MARKERS)
    if spanish == english:
        # Uncountable text - a name, a number, one unfamiliar word. English is
        # the safer default: it is the voice pack with the most options, and a
        # wrong guess there is a different timbre rather than a wrong language.
        return "en"
    return "es" if spanish > english else "en"


SPANISH_MARKERS = (
    "el", "la", "los", "las", "un", "una", "que", "de", "y", "es", "en",
    "por", "con", "para", "del", "se", "su", "más", "pero", "como", "está",
)
ENGLISH_MARKERS = (
    "the", "and", "of", "is", "to", "in", "it", "for", "on", "with", "that",
    "this", "are", "was", "be", "as", "at", "from", "or", "not", "but",
)


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
    """Resolve the wav path inside salida/, which is the only place audio lands.

    A caller that already spelled the `salida/` prefix - the orchestrator does,
    because it resolves the absolute path before calling - must not end up with
    `salida/salida/voz.wav`. So the directory is only prepended when the
    requested name is genuinely relative to the project.
    """
    name = (requested or "").strip()
    if not name:
        name = f"voz-{int(time.time())}.{suffix}"
    elif not Path(name).suffix:
        name = f"{name}.{suffix}"
    path = Path(name)
    if not path.is_absolute():
        # `salida/x.wav` is already the right place; `x.wav` is not.
        if path.parts and path.parts[0] == DEFAULT_OUTPUT_DIR.name:
            path = REPO_ROOT / path
        else:
            path = DEFAULT_OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


VOICE_REPO = "hexgrad/Kokoro-82M"
"""El repositorio de pesos y voces, nombrado en explícito.

Kokoro avisa por stderr si no se pasa, y ese aviso se cuela en la salida del
script junto a la línea `RESULT` que lee el orquestador.
"""


def _cuda_usable() -> bool:
    """Whether CUDA is present and has room for the voice model right now.

    Checked before building the pipeline because `KPipeline(device="cuda")`
    loads the model onto the card immediately, and a card that is already full
    makes that load raise rather than return.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
        # Kokoro son ~330 MB en fp32 más su contexto; 1.5 GB deja margen para
        # que la síntesis no se quede sin sitio a mitad.
        return free > 1_500 * 1024 * 1024
    except Exception:  # noqa: BLE001 - no torch, no driver, no card
        return False


def _is_out_of_memory(error: Exception) -> bool:
    """Whether an exception is a CUDA OOM rather than something else."""
    text = f"{type(error).__name__} {error}".lower()
    return "out of memory" in text or "cuda oom" in text


def _ensure_en_spacy() -> str:
    """Install the English spaCy model if it is missing, or say why it could not be.

    Installs the wheel by URL rather than shelling out to `spacy download`,
    which reported success on this machine without installing anything.
    """
    try:
        import spacy

        # `spacy.load` is the real test. `find_spec` looks for an importable
        # module of that name, and the model is data, not a package, so it
        # reports missing for a model that is installed and working.
        spacy.load(EN_SPACY_MODEL)
        return ""
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - present but unloadable, so reinstall
        pass

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", EN_SPACY_WHEEL],
            capture_output=True, text=True, timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return (
            f"Kokoro necesita {EN_SPACY_MODEL} para hablar inglés y no se pudo instalar: {error}\n"
            f"Instálalo con: {sys.executable} -m pip install '{EN_SPACY_WHEEL}'"
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-2:]
        return (
            f"Kokoro necesita {EN_SPACY_MODEL} para hablar inglés y la instalación falló. "
            f"Instálalo con: {sys.executable} -m pip install '{EN_SPACY_WHEEL}'"
            + (f"\n({' / '.join(detail)})" if detail else "")
        )
    try:
        import spacy

        spacy.load(EN_SPACY_MODEL)
    except Exception as error:  # noqa: BLE001
        return f"{EN_SPACY_MODEL} se instaló pero sigue sin cargar: {error}"
    return ""


def _explain_kokoro_failure(message: str, voice: str) -> str:
    """Turn Kokoro's raw errors into something the model can act on."""
    if "en_core_web_sm" in message or "Can't find model" in message:
        return (
            f"La voz '{voice}' necesita el modelo de spaCy {EN_SPACY_MODEL}, que no está. "
            f"Instálalo con: {sys.executable} -m pip install '{EN_SPACY_WHEEL}' "
            "(o usa una voz en español, que no lo necesita)"
        )
    if "404" in message and "voices/" in message:
        return (
            f"La voz '{voice}' no existe en hexgrad/Kokoro-82M. En español solo está 'ef_dora'; "
            "para inglés, af_bella / af_heart / af_nicole / af_sarah."
        )
    return f"No se pudo cargar Kokoro: {message}"


def speak(text: str, output: str, voice: str, speed: float) -> dict:
    """Synthesize speech with Kokoro and write a single wav."""
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as error:
        return {"ok": False, "error": f"{_KOKORO_HINT} (falta {error.name})"}

    if not voice:
        voice = KOKORO_VOICES.get(detect_language(text), KOKORO_VOICES["en"])
    if voice.startswith("af"):
        failure = _ensure_en_spacy()
        if failure:
            return {"ok": False, "error": failure}

    # El dispositivo se elige al construir el pipeline: `KPipeline` no tiene un
    # `.to()`, y llamarlo es un AttributeError, no una mudanza de sitio.
    #
    # La voz es lo único que tiene que funcionar siempre. Si la GPU está
    # ocupada -un modelo de Ollama resident se lleva 5.5 de los 8 GB-, Kokoro
    # revienta con un OOM a mitad de síntesis y el agente no puede contestar.
    # En CPU tarda más y siempre cabe, así que un OOM en GPU se reintenta allí.
    devices = ["cuda", "cpu"]
    if not _cuda_usable():
        devices = ["cpu"]

    started = time.time()
    chunks: list = []
    last_error = ""
    for device in devices:
        try:
            pipeline = KPipeline(lang_code=voice[0], repo_id=VOICE_REPO, device=device)
        except Exception as error:  # noqa: BLE001 - surfaced to the caller as text
            last_error = _explain_kokoro_failure(str(error), voice)
            continue
        try:
            # Un fragmento por llamada: el contexto de Kokoro es corto, y un
            # párrafo entero sonido como un solo bloque suena a zumbido.
            chunks = []
            for piece in split_for_speech(text):
                chunks.extend(
                    (audio if audio is not None else np.zeros(1))
                    for _, _, audio in pipeline(piece, voice=voice, speed=speed)
                )
        except Exception as error:  # noqa: BLE001
            last_error = f"Kokoro falló sintetizando en {device}: {error}"
            chunks = []
            if not _is_out_of_memory(error):
                return {"ok": False, "error": last_error}
            continue
        if chunks:
            break
    if not chunks:
        return {"ok": False, "error": last_error or "Kokoro no produjo audio para ese texto."}

    if not chunks:
        return {"ok": False, "error": "Kokoro no produjo audio para ese texto."}
    audio = np.concatenate(chunks, axis=0)
    path = _output_path(output, "wav")
    sf.write(str(path), audio, 24000)
    result = {
        "ok": True,
        "path": str(path),
        "voice": voice,
        "seconds": round(len(audio) / 24000, 2),
        "elapsed": round(time.time() - started, 2),
    }
    if result["seconds"] == 0.0:
        return {"ok": False, "error": "Kokoro no produjo audio para ese texto."}
    return result


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


def _whisper_has_gpu(binary: str) -> bool:
    """Whether this whisper-cli build has a `--no-gpu` flag, i.e. was built with CUDA.

    Asked of the binary rather than guessed from the machine: the two answers
    disagree on exactly the setup that matters here, where the card has CUDA
    but nvcc is absent, so the build is CPU-only while the machine looks like
    it has a GPU.
    """
    try:
        completed = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "-ng, --no-gpu" in (completed.stdout + completed.stderr)


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

    model_dir = os.environ.get("WHISPER_MODEL_DIR", "").strip() or str(
        Path.home() / ".cache" / "whisper"
    )
    if not Path(model_dir, f"ggml-{model}.bin").is_file():
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
    # Nada de GPU aquí. whisper.cpp usa la GPU por defecto cuando el binario se
    # compiló con CUDA, y en ese caso compite por la VRAM con el trabajo pesado
    # en curso, que es justo lo que la reserva de voz evita. Se usa el flag
    # explícito en lugar de decidirlo por la presencia de nvidia-smi: en una
    # tarjeta con CUDA pero sin toolkit de compilación, como esta, el binario
    # es de CPU y `-ng` no existe en él, se ignora, y el resultado es que
    # whisper devuelve un wav vacío sin decir por qué.
    if _whisper_has_gpu(binary):
        command.append("-ng")

    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "whisper.cpp tardó más de 10 minutos y se detuvo."}
    except OSError as error:
        return {"ok": False, "error": f"No se pudo ejecutar whisper.cpp: {error}"}

    # whisper.cpp escribe el .txt donde se le dice con -of, y ahí se le pasa la
    # ruta SIN extensión: `salida/audio.wav` produce `salida/audio.txt`, no
    # `salida/audio.wav.txt`. Buscar con la extensión puesta hacía que el
    # script no encontrara su propia salida y reportara que no hubo
    # transcripción, cuando la había y estaba a un `with_suffix` de distancia.
    transcript = Path(audio_path).with_suffix(".txt")
    text = transcript.read_text(encoding="utf-8", errors="replace") if transcript.is_file() else ""
    if not text.strip():
        # Borra el .txt vacío que whisper deja cuando no reconoce nada: si no,
        # el siguiente intento lee este resultado vacío en vez de transcribir.
        transcript.unlink(missing_ok=True)
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        return {
            "ok": False,
            "error": (
                "whisper.cpp no devolvió transcripción. Puede ser que el audio no tenga voz "
                "(solo música o silencio), o que el idioma no sea el indicado.\n" + "\n".join(tail)
            ),
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

#!/usr/bin/env python3
"""Prueba de extremo a extremo de la generación de imágenes.

No está en la suite de pytest a propósito: necesita la GPU casi entera y tarda
un par de minutos, mientras que el resto de los tests corre en seis segundos.
Ejecútala cuando cambie algo del generador, no en cadacommit.

    python scripts/test_imagen.py

Sale con 0 si la imagen se generó y es válida, con 1 si algo falla. Acepta un
prompt propio como argumento:

    python scripts/test_imagen.py "un cocodrilo de goma verde"

Y con --verificar solo comprueba una imagen que ya existe, sin generar nada,
que es lo único posible mientras otra sesión tenga la GPU ocupada:

    python scripts/test_imagen.py --verificar salida/cocodrilo-goma.png
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "scripts" / "generar_imagen.py"
VENV_PYTHON = Path.home() / "qwenimg-venv" / "bin" / "python"
OUTPUT_DIRNAME = "salida"

DEFAULT_PROMPT = "un cocodrilo de goma verde de juguete, foto de producto sobre fondo blanco"
STEPS = 24
MIN_BYTES = 20_000
# Share of the frame that has to differ from its dominant colour. A run that
# half works leaves a flat rectangle, and a file existing is no evidence it did
# not. Measured against the background rather than as global variance, because
# "a green crocodile on white" is 84% white and scores badly on variance while
# being a perfectly good picture.
MIN_SUBJECT_SHARE = 0.02
BACKGROUND_TOLERANCE = 60
SAMPLE_EDGE = 256


def subject_share(path: Path) -> float:
    """How much of the frame is not the background colour, between 0 and 1.

    Down sampled first: at 256x256 the colour histogram is small enough to be
    exact, and what it measures - is there a subject in the picture - does not
    need full resolution.
    """
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("RGB").resize((SAMPLE_EDGE, SAMPLE_EDGE))
    colors = small.getcolors(maxcolors=SAMPLE_EDGE * SAMPLE_EDGE) or []
    if not colors:
        return 0.0
    total = sum(count for count, _ in colors)
    dominant = max(colors, key=lambda entry: entry[0])[1]
    apart = 0
    for count, (red, green, blue) in colors:
        if abs(red - dominant[0]) + abs(green - dominant[1]) + abs(blue - dominant[2]) > BACKGROUND_TOLERANCE:
            apart += count
    return apart / total


def check_image(path: Path) -> str | None:
    """Why this file is not a usable image, or None when it is."""
    if not path.is_file():
        return f"no existe {path}"
    if not path.is_relative_to(REPO_ROOT / OUTPUT_DIRNAME):
        return f"cae fuera de {OUTPUT_DIRNAME}/: {path}"
    data = path.read_bytes()
    if len(data) < MIN_BYTES:
        return f"son {len(data)} bytes; un PNG de 1024x1024 pesa bastante mas"
    try:
        import PIL  # noqa: F401
    except ImportError:
        return None  # Without Pillow the file's existence and size are all there is to say.
    share = subject_share(path)
    if share < MIN_SUBJECT_SHARE:
        return (
            f"solo el {share * 100:.1f}% de la imagen se aparta del fondo; "
            f"hace falta un {MIN_SUBJECT_SHARE * 100:.0f}%. Parece un rectángulo plano."
        )
    return None


def describe(path: Path, elapsed: float) -> str:
    from PIL import Image

    data = path.read_bytes()
    with Image.open(path) as image:
        size = image.size
    return json.dumps(
        {
            "resultado": "OK",
            "fichero": f"{OUTPUT_DIRNAME}/{path.name}",
            "bytes": len(data),
            "tamano": f"{size[0]}x{size[1]}",
            "sujeto_sobre_fondo": f"{subject_share(path) * 100:.1f}%",
            "segundos": round(elapsed, 1),
        },
        indent=2,
    )


def free_vram_mib() -> float | None:
    """VRAM libre, or None when nvidia-smi is not there to answer."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return float(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments and arguments[0] == "--verificar":
        if len(arguments) < 2:
            print("FALLO: --verificar necesita la ruta de una imagen")
            return 1
        candidate = Path(arguments[1])
        path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        problem = check_image(path)
        if problem:
            print(f"FALLO: {problem}")
            return 1
        print(describe(path, 0.0))
        return 0

    prompt = arguments[0] if arguments else DEFAULT_PROMPT

    if not VENV_PYTHON.exists():
        print(f"FALLO: no existe el interprete {VENV_PYTHON}")
        return 1

    free = free_vram_mib()
    if free is not None and free < 6000:
        # The denoiser plus the VAE need the card. Saying so up front beats a
        # CUDA out of memory a minute later, in the middle of the diffusion.
        print(f"FALLO: solo hay {free:.0f} MiB de VRAM libre; hacen falta unos 6000.")
        print("       El denoiser y el VAE no caben con el modelo de lenguaje ya cargado.")
        print("       Suelta el LLM (en ollama: 'ollama stop <modelo>') y vuelve a intentarlo.")
        return 1

    before = {p.name for p in (REPO_ROOT / OUTPUT_DIRNAME).glob("*.png")}
    started = time.perf_counter()
    # cwd is deliberately the repo, and the point is that the destination does
    # not depend on it: --output-dir is a relative path.
    completed = subprocess.run(
        [str(VENV_PYTHON), str(GENERATOR), prompt, "--steps", str(STEPS), "--name", "test-cocodrilo"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        print("FALLO: el generador devolvio", completed.returncode)
        print(output[-2000:])
        return 1

    after = {p.name for p in (REPO_ROOT / OUTPUT_DIRNAME).glob("*.png")}
    created = sorted(after - before)
    if not created:
        print("FALLO: el generador termino bien pero no creo ninguna imagen nueva")
        print(output[-2000:])
        return 1
    path = REPO_ROOT / OUTPUT_DIRNAME / created[0]

    problem = check_image(path)
    if problem:
        print(f"FALLO: {problem}")
        return 1
    print(describe(path, elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

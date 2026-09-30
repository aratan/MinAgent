"""Contracts of the music script that hold without loading a model.

Everything here runs offline. The point of each test is a mistake that a full
generation would hide: a path that lands in the wrong directory, a duration that
becomes the wrong number of tokens, a silent clip reported as a success. A
generation costs 10 s and 1300 MiB, so none of these failures can be found by
running the happy path - which is exactly why they need their own tests.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _musica():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("musica_script", root / "scripts" / "compute" / "musica.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(capsys) -> dict:
    """Read the one RESULT line the script prints."""
    out = capsys.readouterr().out.strip()
    assert out.startswith("RESULT "), out
    return json.loads(out[len("RESULT ") :])


def test_estado_works_without_importing_the_model_stack():
    """`--estado` has to answer on a machine that cannot generate music.

    It is the call a diagnostic makes to find out whether music is available, so
    it must not need CUDA, nor transformers, nor the weights. Importing torch
    eagerly to answer it would make the diagnostic itself the thing that fails -
    and a diagnostic that fails is read as "music unavailable", which is a
    different and wrong answer. Run in a clean interpreter so the assertion is
    about this script and not about some earlier test having imported torch.
    """
    root = Path(__file__).resolve().parent.parent
    probe = (
        "import importlib.util, json, sys, io, contextlib\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(root / 'scripts' / 'compute' / 'musica.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    assert module.main(['--estado']) == 0\n"
        "print('LOADED:' + ','.join(name for name in ('torch', 'transformers', 'soundfile') if name in sys.modules))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, cwd=str(root)
    )

    assert finished.returncode == 0, finished.stderr
    loaded = finished.stdout.strip().splitlines()[-1]
    assert loaded == "LOADED:", f"--estado pulled in {loaded[len('LOADED:'):]}"


def test_estado_reports_the_only_model_it_offers(musica_module, capsys):
    """The report has to describe what is actually there.

    It listed three models once, when the script had three names to offer and
    two of them could not run. A report that overstates capability is worse than
    no report: it is read as permission to try.
    """
    musica_module.main(["--estado"])

    payload = _payload(capsys)
    assert payload["modelos"] == ["small"]
    assert payload["modelo_defecto"] == "small"
    # The number has to be the one the VRAM check will compare against.
    assert payload["vram_estimada_mib"] == musica_module.VRAM_ESTIMATE_MIB["small"]
    assert payload["max_segundos"] == musica_module.MAX_SECONDS


def test_a_model_that_is_not_offered_is_refused(musica_module, capsys):
    """Naming a model that does not exist must fail, not fall back silently.

    Falling back to the one model that works would hide the mistake, and the
    caller would get audio from a setting they never asked for.
    """
    with pytest.raises(SystemExit):
        musica_module.main(["--model", "music", "--prompt", "jazz"])

    assert "small" in capsys.readouterr().err


def test_duration_becomes_tokens_from_the_model_itself(musica_module):
    """Tokens come from the model's own frame rate, not a constant in this file.

    The 50 tokens per second figure is what the config said at the time; if the
    model ever changes its rate, a hardcoded number would quietly produce a clip
    of the wrong length instead of an error.
    """
    config = SimpleNamespace(audio_encoder=SimpleNamespace(hop_length=320), frame_rate=50.0)

    assert musica_module._token_count(config, 5) == 250
    assert musica_module._token_count(config, 30) == 1500


def test_a_rate_of_zero_does_not_divide_by_zero(musica_module):
    """A config with no frame rate must not crash the length calculation.

    It should fall back to the documented rate, because a crash here happens
    after the model is already loaded, where it costs the most.
    """
    config = SimpleNamespace(audio_encoder=SimpleNamespace(hop_length=320), frame_rate=0.0)

    assert musica_module._token_count(config, 5) == 250


@pytest.mark.parametrize(
    ("requested", "expected_tail"),
    [
        ("clip", "salida/clip.wav"),
        ("clip.wav", "salida/clip.wav"),
        ("salida/clip.wav", "salida/clip.wav"),
    ],
)
def test_output_lands_in_the_audio_directory_with_a_wav_extension(musica_module, requested, expected_tail):
    """A name without an extension becomes a wav, and it lands in `salida/`.

    Two mistakes are being locked at once. A bare name written to the working
    directory is invisible to the rest of MinAgent, which only looks in
    `salida/`; and a file with no extension is one the player cannot open.
    """
    path = musica_module._output_path(requested, "clip")

    assert path.suffix == ".wav"
    assert path.as_posix().endswith(expected_tail)


def test_an_explicit_directory_is_not_rewritten(musica_module, tmp_path):
    """A path the caller spelled out in full is left exactly as given."""
    path = musica_module._output_path(str(tmp_path / "mine.wav"), "clip")

    assert path == tmp_path / "mine.wav"


def test_digital_silence_is_not_reported_as_music(musica_module):
    """A silent clip is a failure, not a short track.

    The model can return silence without raising anything, and a file that
    exists and is inaudible looks exactly like a success to everything
    downstream. Only the signal itself can tell them apart.
    """
    assert musica_module._is_silent(np.zeros(1000, dtype=np.float32)) is True
    assert musica_module._is_silent(np.full(1000, 0.2, dtype=np.float32)) is False


def test_conversion_off_the_cuda_tensor_is_numpy_float(musica_module):
    """What the model returns is an unbatched CUDA fp16 tensor; soundfile needs
    CPU float32 samples.

    numpy has no float16, and a CUDA tensor cannot be indexed by numpy at all.
    Without all three steps in order this raises instead of writing a file, and
    it raises after the generation that cost the time. Indexing `[0, 0]` is the
    step that picks the one channel out of the `[batch, channels, samples]`
    shape the decoder emits.
    """
    class _FakeTensor:
        def __getitem__(self, index):
            assert index == (0, 0), "only the unbatched mono case is modelled here"
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 4), dtype=np.float32)

    samples = musica_module._to_samples(_FakeTensor())

    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    # Flattened, because reshape(-1) is what turns [1, 4] into four samples.
    assert samples.shape == (4,)


@pytest.fixture
def musica_module():
    return _musica()

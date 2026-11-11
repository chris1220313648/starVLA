"""Check checkpoint frequency reaches the trainer without launching training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/simBenchmarks/Robotwin/train_files/run_groundingpi_pi05_pipeline.py"


@pytest.mark.parametrize("interval", [None, 5000])
def test_save_interval_and_smoke(tmp_path, monkeypatch, interval):
    spec = importlib.util.spec_from_file_location("pi05_pipeline", SCRIPT)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    argv = [str(SCRIPT), "--output", str(tmp_path / "run"), "--color-jitter", "0"]
    if interval is not None:
        argv += ["--save-interval", str(interval)]
    monkeypatch.setattr(pipeline.sys, "argv", argv)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(pipeline.os, "killpg", lambda *a: None)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *a: None)
    commands = []

    class Process:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self, **kwargs):
            return 0 if len(commands) == 1 else 1

        def poll(self):
            return 1

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def add(self, *args, **kwargs):
            pass

    monkeypatch.setattr(pipeline.tarfile, "open", lambda *a, **k: Archive())
    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="Smoke training failed"):
        pipeline.main()
    assert commands[0][commands[0].index("--save-interval") + 1] == str(interval or 1000)
    assert commands[1][commands[1].index("--save-interval") + 1] == "3"


@pytest.mark.parametrize("interval", ["0", "-1"])
def test_invalid_save_interval(tmp_path, monkeypatch, interval):
    spec = importlib.util.spec_from_file_location("pi05_pipeline", SCRIPT)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    output = tmp_path / "run"
    monkeypatch.setattr(pipeline.sys, "argv", [str(SCRIPT), "--output", str(output),
                                            "--color-jitter", "0", "--save-interval", interval])
    with pytest.raises(SystemExit) as error:
        pipeline.main()
    assert error.value.code == 2
    assert not output.exists()

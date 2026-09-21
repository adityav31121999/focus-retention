import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_accelerator as launcher


@pytest.mark.parametrize("accelerator,devices", [("gpu", 1), ("gpu", 2), ("tpu", 8)])
def test_routes_to_device_script(accelerator, devices):
    command = launcher.build_command(dict(accelerator=accelerator, devices=devices), ["--data", "tokens.bin"])
    assert str(launcher.ROOT / "scripts" / f"train_{accelerator}.py") in command
    assert ("torch.distributed.run" in command) == (accelerator == "gpu" and devices > 1)
    if accelerator == "gpu" and devices > 1:
        assert f"--nproc_per_node={devices}" in command
    if accelerator == "tpu":
        assert command[command.index("--expected-devices") + 1] == "8"
    config = command[command.index("--config") + 1]
    assert Path(config).name == ("curriculum_tpu.yaml" if accelerator == "tpu" else "curriculum_kaggle.yaml")


def test_preserves_explicit_trainer_options():
    command = launcher.build_command(dict(accelerator="tpu", devices=8),
        ["--config=custom.yaml", "--expected-devices=4", "--seq-len", "512", "--synthetic"])
    assert "--config" not in command
    assert "--expected-devices" not in command
    assert "--config=custom.yaml" in command
    assert command[command.index("--seq-len") + 1] == "512"


def test_detection_uses_child_process(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="", stdout='initialization log\nHARDWARE_JSON={"accelerator":"tpu","devices":8}\n')
    monkeypatch.setattr(launcher.subprocess, "run", run)
    result = launcher.detect_accelerator("tpu")
    assert result == dict(accelerator="tpu", devices=8)
    assert calls[0][0][1] == "-c"
    assert calls[0][1]["env"]["PJRT_DEVICE"] == "TPU"


def test_failed_detection_never_starts_training(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    def detect(_):
        raise RuntimeError("No accelerator")
    monkeypatch.setattr(launcher, "detect_accelerator", detect)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected trainer process"))
    assert launcher.main(["--synthetic"]) == 1


def test_tpu_launch_sets_environment_and_propagates_exit_code(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(launcher, "detect_accelerator", lambda _: dict(accelerator="tpu", devices=8))
    def run(command, **kwargs):
        assert kwargs["env"]["PJRT_DEVICE"] == "TPU"
        assert str(launcher.ROOT / "scripts" / "train_tpu.py") in command
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(["--synthetic", "--tiny"]) == 7


def test_detect_only_outputs_json(monkeypatch, capsys):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(launcher, "detect_accelerator", lambda _: dict(accelerator="gpu", devices=2))
    assert launcher.main(["--detect-only"]) == 0
    assert json.loads(capsys.readouterr().out) == dict(accelerator="gpu", devices=2)

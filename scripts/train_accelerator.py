"""Detect hardware in a disposable process and launch the corresponding trainer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = r"""
import importlib.util
import json
import sys
import torch

requested = sys.argv[1]
try:
    gpu_count = torch.cuda.device_count() if requested != 'tpu' else 0
    if requested == 'gpu' or (requested == 'auto' and gpu_count):
        if not gpu_count:
            raise RuntimeError('No CUDA GPU is visible. Select a GPU accelerator in Kaggle.')
        info = dict(accelerator='gpu', devices=gpu_count, torch=torch.__version__,
                    names=[torch.cuda.get_device_name(i) for i in range(gpu_count)])
    else:
        if importlib.util.find_spec('torch_xla') is None:
            raise RuntimeError('No CUDA GPU or torch-xla installation found. Select a GPU or TPU accelerator.')
        import torch_xla
        import torch_xla.runtime as xr
        if xr.device_type() != 'TPU':
            raise RuntimeError('The XLA runtime is not a TPU runtime.')
        if torch.__version__.split('+')[0].split('.')[:2] != torch_xla.__version__.split('+')[0].split('.')[:2]:
            raise RuntimeError('PyTorch and torch-xla major/minor versions must match.')
        torch_xla.device()
        info = dict(accelerator='tpu', devices=xr.addressable_device_count(),
                    torch=torch.__version__, torch_xla=torch_xla.__version__)
        if info['devices'] < 1:
            raise RuntimeError('No TPU devices are addressable.')
    print('HARDWARE_JSON=' + json.dumps(info))
except Exception as error:
    print('Hardware detection failed: ' + str(error), file=sys.stderr)
    sys.exit(1)
"""


def detect_accelerator(requested="auto"):
    environment = os.environ.copy()
    if requested == "tpu":
        environment["PJRT_DEVICE"] = "TPU"
    elif requested == "auto":
        environment.setdefault("PJRT_DEVICE", "TPU")
    result = subprocess.run([sys.executable, "-c", PROBE, requested], env=environment,
                            capture_output=True, text=True, timeout=120)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError("Accelerator detection failed; see the diagnostic above. No training process was started.")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("HARDWARE_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("Hardware probe returned no accelerator information")


def has_option(arguments, name):
    return any(value == name or value.startswith(name + "=") for value in arguments)


def build_command(hardware, training_arguments):
    arguments = list(training_arguments)
    accelerator, devices = hardware["accelerator"], hardware["devices"]
    if devices < 1 or accelerator not in ("gpu", "tpu"):
        raise ValueError("Expected at least one GPU or TPU device")
    if not has_option(arguments, "--config"):
        name = "curriculum_tpu.yaml" if accelerator == "tpu" else "curriculum_kaggle.yaml"
        arguments += ["--config", str(ROOT / "configs" / name)]
    script = str(ROOT / "scripts" / f"train_{accelerator}.py")
    if accelerator == "gpu" and devices > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={devices}", script] + arguments
    if accelerator == "tpu" and not has_option(arguments, "--expected-devices"):
        arguments += ["--expected-devices", str(devices)]
    return [sys.executable, script] + arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Pass trainer options after the launcher options, e.g. --data tokens.bin --seq-len 128. "
               "GPU/TPU-specific options are documented in scripts/TRAINING.md.")
    parser.add_argument("--accelerator", choices=("auto", "gpu", "tpu"), default="auto")
    parser.add_argument("--detect-only", action="store_true", help="Print hardware JSON and exit")
    parser.add_argument("--dry-run", action="store_true", help="Detect hardware and print the selected command")
    args, training_arguments = parser.parse_known_args(argv)
    if "RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Run this launcher with python, not torchrun; it starts the device workers itself")
    try:
        hardware = detect_accelerator(args.accelerator)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.detect_only:
        print(json.dumps(hardware))
        return 0
    command = build_command(hardware, training_arguments)
    print(f"Detected {hardware['accelerator'].upper()}: {hardware['devices']} device(s)", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if hardware["accelerator"] == "tpu":
        environment["PJRT_DEVICE"] = "TPU"
    # Preserve the caller's working directory so relative data paths remain valid.
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())

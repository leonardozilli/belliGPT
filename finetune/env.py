import os
from pathlib import Path
from typing import Literal

Environment = Literal["local", "colab", "remote"]

DRIVE_ROOT = Path("/content/drive/MyDrive/belligpt")


def in_colab() -> bool:
    return any(k.startswith("COLAB_") for k in os.environ)


def detect_environment() -> Environment:
    if in_colab():
        return "colab"
    if os.environ.get("FT_ENV", "").lower() == "remote":
        return "remote"
    return "local"


def repo_root() -> Path:
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent.parent


def storage_roots(env: Environment) -> tuple[Path, Path]:
    if env == "colab":
        return DRIVE_ROOT / "finetune/data", DRIVE_ROOT / "finetune/outputs"
    base = repo_root() / "finetune"
    return base / "data", base / "outputs"


def gpu_memory_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:
        return None


def auto_batch(target_effective: int = 64) -> tuple[int, int]:
    """Pick (per_device_batch_size, grad_accum) based on the detected VRAM."""
    mem = gpu_memory_gb()
    if mem is None:
        per_device = 2
    elif mem < 12:
        per_device = 16
    elif mem < 24:
        per_device = 32
    elif mem < 48:
        per_device = 64
    else:
        per_device = 64
    per_device = min(per_device, target_effective)
    grad_accum = max(1, target_effective // per_device)
    return per_device, grad_accum


def describe() -> str:
    env = detect_environment()
    mem = gpu_memory_gb()
    gpu = f"{mem:.0f} GB GPU" if mem else "no GPU"
    return f"env={env} | {gpu}"

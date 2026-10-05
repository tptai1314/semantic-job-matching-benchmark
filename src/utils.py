from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from src.config import get


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Ghim seed cho random, numpy, torch (và PYTHONHASHSEED cho tiến trình con)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # pragma: no cover - dùng cho DataLoader
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(cfg: dict[str, Any]) -> str:
    requested = get(cfg, "runtime.device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested

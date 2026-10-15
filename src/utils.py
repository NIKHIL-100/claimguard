"""Shared helpers: config loading and reproducible seeding."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = PROJECT_ROOT / "configs" / "config.yaml") -> dict:
    """Read the YAML config so every script uses the same paths and settings."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix every source of randomness.

    Why: if splits or weight initialisation change between runs, you can't tell
    whether a metric improved because of your change or because of luck.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is only needed from Step 5 onwards
        pass

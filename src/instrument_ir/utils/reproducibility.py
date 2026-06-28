"""Determinismo y semillas (ADR §8.1)."""

from __future__ import annotations

import os
import random


def set_global_determinism(seed: int = 42) -> None:
    """Fija semillas en random/numpy/torch y flags deterministas de cuDNN si torch está disponible."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy siempre está, pero no forzamos torch
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch es opcional (extra [dense]); en la capa de datos no hace falta.
        pass

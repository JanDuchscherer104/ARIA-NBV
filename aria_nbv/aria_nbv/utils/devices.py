"""Shared device and accelerator selectors for Aria-NBV configs."""

from __future__ import annotations

from enum import StrEnum

import torch


class TorchAccelerator(StrEnum):
    """Global torch accelerator selector used by profile-level configs."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"


def is_mps_available() -> bool:
    """Return True when Torch reports a usable MPS backend."""

    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


__all__ = ["TorchAccelerator", "is_mps_available"]

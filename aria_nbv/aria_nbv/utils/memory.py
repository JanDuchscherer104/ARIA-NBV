"""Shared memory-footprint helpers for tensor-heavy diagnostics.

This module owns best-effort size estimation utilities for tensors, NumPy
arrays, and nested container structures so app panels and diagnostics can reuse
one canonical implementation.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import torch


def tensor_nbytes(value: torch.Tensor) -> int:
    """Return the raw byte size of a tensor storage view."""

    return int(value.numel()) * int(value.element_size())


def estimate_nbytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Best-effort estimate of nested tensor-container memory footprint."""

    if value is None:
        return 0

    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    if torch.is_tensor(value):
        return tensor_nbytes(value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    tensor_fn = getattr(value, "tensor", None)
    if callable(tensor_fn):
        try:
            tensor = tensor_fn()
        except Exception:
            tensor = None
        if torch.is_tensor(tensor):
            return tensor_nbytes(tensor)

    if is_dataclass(value):
        return sum(estimate_nbytes(getattr(value, field.name), _seen=_seen) for field in fields(value))
    if isinstance(value, dict):
        return sum(estimate_nbytes(item, _seen=_seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(estimate_nbytes(item, _seen=_seen) for item in value)
    if hasattr(value, "__dict__"):
        return sum(estimate_nbytes(item, _seen=_seen) for item in vars(value).values())
    return 0


def p3d_cameras_nbytes(value: Any) -> int:
    """Estimate size of the commonly-used `PerspectiveCameras` tensor fields."""

    if value is None:
        return 0
    total = 0
    for name in ("R", "T", "focal_length", "principal_point", "image_size"):
        tensor = getattr(value, name, None)
        if torch.is_tensor(tensor):
            total += tensor_nbytes(tensor)
    return total


__all__ = ["estimate_nbytes", "p3d_cameras_nbytes", "tensor_nbytes"]

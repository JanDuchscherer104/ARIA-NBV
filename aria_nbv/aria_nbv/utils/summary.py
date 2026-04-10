from typing import Any

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from torch import Tensor


def summarize(
    val: Tensor | CameraTW | PoseTW | ObbTW | TensorWrapper | list | Any, *, include_stats: bool = False
) -> Any:
    """Small helper for succinct repr output."""
    if val is None:
        return None
    if isinstance(val, list):
        return {"len": len(val)}
    tensor = _extract_tensor(val)
    if tensor is not None:
        return _tensor_summary(tensor, include_stats=include_stats)
    return val


def _extract_tensor(val: Any) -> Tensor | None:
    """Return a tensor view for common tensor-wrapper types when possible."""
    if isinstance(val, Tensor):
        return val
    if isinstance(val, PoseTW):
        return val.matrix
    if isinstance(val, (TensorWrapper, CameraTW, ObbTW)):
        data = val.tensor() if callable(getattr(val, "tensor", None)) else val.tensor  # type: ignore[operator]
        return data
    return None


def _tensor_summary(tensor: Tensor, *, include_stats: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    if tensor.numel() and torch.is_floating_point(tensor) and include_stats:
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel():
            summary["min"] = float(finite.min())
            summary["max"] = float(finite.max())
            summary["mean"] = float(finite.mean())
    return summary

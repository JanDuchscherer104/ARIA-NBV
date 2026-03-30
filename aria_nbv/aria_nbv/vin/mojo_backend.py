"""Mojo-backed CPU accumulation kernels for VIN semidense projection features.

This module keeps the existing PyTorch/PyTorch3D path intact and only activates
when ``SemidenseProjectionBackend.MOJO`` is selected. The Python side stages
contiguous CPU buffers and hands raw addresses to a lazily imported Mojo module.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

_MOJO_KERNEL_MODULE = "vin_projection_kernels"
_MOJO_SITE_PACKAGES_ENV = "ARIA_NBV_MOJO_SITE_PACKAGES"
_MOJO_THREADS_ENV = "ARIA_NBV_MOJO_THREADS"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parent / "mojo"


def _candidate_site_packages() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get(_MOJO_SITE_PACKAGES_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(sorted((_repo_root() / ".mojo-venv" / "lib").glob("python*/site-packages")))
    return candidates


def _has_mojo_importer() -> bool:
    try:
        return importlib.util.find_spec("mojo.importer") is not None
    except ModuleNotFoundError:
        return False


def _ensure_mojo_importer() -> None:
    if _has_mojo_importer():
        return

    for candidate in _candidate_site_packages():
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        if _has_mojo_importer():
            return

    raise ModuleNotFoundError(
        "Mojo Python importer not found. Install Mojo into `<repo>/.mojo-venv` "
        f"or set `{_MOJO_SITE_PACKAGES_ENV}` to the matching site-packages path."
    )


@lru_cache(maxsize=1)
def _load_mojo_kernels() -> Any:
    _ensure_mojo_importer()
    import mojo.importer  # noqa: F401

    kernel_dir = _kernel_dir()
    kernel_dir_str = str(kernel_dir)
    if kernel_dir_str not in sys.path:
        sys.path.insert(0, kernel_dir_str)

    importlib.invalidate_caches()
    return importlib.import_module(_MOJO_KERNEL_MODULE)


def is_mojo_available() -> bool:
    """Return ``True`` when the Mojo runtime and local kernels can be imported."""

    try:
        _load_mojo_kernels()
    except Exception:
        return False
    return True


def _resolve_workers(num_items: int, requested: int | None = None) -> int:
    if requested is not None:
        workers = int(requested)
    else:
        env_value = os.environ.get(_MOJO_THREADS_ENV)
        workers = int(env_value) if env_value is not None else (os.cpu_count() or 1)
    workers = max(1, workers)
    return max(1, min(workers, max(1, int(num_items))))


def _to_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _to_cpu_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _tensor_numpy_view(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.numpy()  # type: ignore[no-any-return]


def accumulate_projection_bins_mojo(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    finite: torch.Tensor,
    valid: torch.Tensor,
    w_rel: torch.Tensor,
    image_size: torch.Tensor,
    grid_size: int,
    device: torch.device,
    workers: int | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate semidense projection bins with the experimental Mojo reducer."""

    x_cpu = _to_cpu_float32(x)
    y_cpu = _to_cpu_float32(y)
    z_cpu = _to_cpu_float32(z)
    finite_cpu = _to_cpu_uint8(finite)
    valid_cpu = _to_cpu_uint8(valid)
    w_rel_cpu = _to_cpu_float32(w_rel)
    image_size_cpu = _to_cpu_float32(image_size.view(-1, 2))

    num_cams, num_points = x_cpu.shape
    num_bins = int(grid_size) * int(grid_size)

    x_np = _tensor_numpy_view(x_cpu)
    y_np = _tensor_numpy_view(y_cpu)
    z_np = _tensor_numpy_view(z_cpu)
    finite_np = _tensor_numpy_view(finite_cpu)
    valid_np = _tensor_numpy_view(valid_cpu)
    w_rel_np = _tensor_numpy_view(w_rel_cpu)
    image_size_np = _tensor_numpy_view(image_size_cpu)

    counts = np.zeros((num_cams, num_bins), dtype=np.float32)
    sum_z = np.zeros_like(counts)
    sum_z2 = np.zeros_like(counts)
    weight_valid_sum = np.zeros(num_cams, dtype=np.float32)
    weight_finite_sum = np.zeros(num_cams, dtype=np.float32)
    weight_z_sum = np.zeros(num_cams, dtype=np.float32)
    weight_z2_sum = np.zeros(num_cams, dtype=np.float32)

    kernels = _load_mojo_kernels()
    kernels.accumulate_projection_bins_f32(
        x_np.ctypes.data,
        y_np.ctypes.data,
        z_np.ctypes.data,
        (
            finite_np.ctypes.data,
            valid_np.ctypes.data,
            w_rel_np.ctypes.data,
        ),
        (
            image_size_np.ctypes.data,
            num_cams,
            num_points,
            int(grid_size),
            _resolve_workers(num_cams, workers),
        ),
        (
            counts.ctypes.data,
            sum_z.ctypes.data,
            sum_z2.ctypes.data,
            weight_valid_sum.ctypes.data,
            weight_finite_sum.ctypes.data,
            weight_z_sum.ctypes.data,
            weight_z2_sum.ctypes.data,
        ),
    )

    def _to_device(array: np.ndarray[Any, Any]) -> torch.Tensor:
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    return {
        "counts": _to_device(counts),
        "sum_z": _to_device(sum_z),
        "sum_z2": _to_device(sum_z2),
        "weight_valid_sum": _to_device(weight_valid_sum),
        "weight_finite_sum": _to_device(weight_finite_sum),
        "weight_z_sum": _to_device(weight_z_sum),
        "weight_z2_sum": _to_device(weight_z2_sum),
    }


__all__ = ["accumulate_projection_bins_mojo", "is_mojo_available"]

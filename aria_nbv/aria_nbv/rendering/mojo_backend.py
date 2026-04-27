"""Mojo-backed rendering kernels for oracle depth and point-cloud stages."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from efm3d.aria import CameraTW, PoseTW

_MOJO_KERNEL_MODULE = "oracle_render_kernels"
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
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    lib_dir = _repo_root() / ".mojo-venv" / "lib"
    exact_matches = sorted(lib_dir.glob(f"{version_dir}/site-packages"))
    candidates.extend(exact_matches)
    candidates.extend(path for path in sorted(lib_dir.glob("python*/site-packages")) if path not in exact_matches)
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
            sys.path.append(candidate_str)
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
    """Return ``True`` when the Mojo runtime and local kernels import cleanly."""

    try:
        _load_mojo_kernels()
    except Exception:
        return False
    return True


def is_mojo_thread_context_supported() -> bool:
    """Return True when Python-imported Mojo render kernels may run in this thread."""

    return threading.current_thread() is threading.main_thread()


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


def _camera_parameters(camera: CameraTW) -> tuple[int, int, float, float, float, float]:
    size = camera.size.reshape(-1, 2)[0].detach().cpu()
    focals = camera.f.reshape(-1, 2)[0].detach().cpu()
    centers = camera.c.reshape(-1, 2)[0].detach().cpu()
    return (
        int(size[0].item()),
        int(size[1].item()),
        float(focals[0].item()),
        float(focals[1].item()),
        float(centers[0].item()),
        float(centers[1].item()),
    )


def _pose_matrix3x4(pose_world_cam: PoseTW) -> torch.Tensor:
    matrix = pose_world_cam.matrix
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix[:3, :4].detach().to(dtype=torch.float32).contiguous()


def render_depth_map_mojo(
    triangles_cam: torch.Tensor,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    znear: float,
    zfar: float,
    device: torch.device,
    workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one depth map using the Mojo closest-hit kernel."""

    triangles_cpu = _to_cpu_float32(triangles_cam.reshape(-1, 3, 3))
    triangles_np = _tensor_numpy_view(triangles_cpu)
    num_pixels = int(width) * int(height)
    depth = np.empty(num_pixels, dtype=np.float32)
    hit = np.empty(num_pixels, dtype=np.uint8)

    kernels = _load_mojo_kernels()
    kernels.render_depth_map_f32(
        triangles_np.ctypes.data,
        triangles_cpu.shape[0],
        (
            int(width),
            int(height),
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            float(znear),
            float(zfar),
            _resolve_workers(num_pixels, workers),
        ),
        depth.ctypes.data,
        hit.ctypes.data,
    )
    depth_t = torch.from_numpy(depth.reshape(height, width)).to(device=device, dtype=torch.float32)
    hit_t = torch.from_numpy(hit.reshape(height, width).astype(np.bool_, copy=False)).to(device=device)
    return depth_t, hit_t


def unproject_candidate_points_mojo(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pose_world_cam: PoseTW,
    camera: CameraTW,
    stride: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backproject one candidate depth map with stable compaction order."""

    width, height, fx, fy, cx, cy = _camera_parameters(camera)
    depth_cpu = _to_cpu_float32(depth.reshape(-1))
    valid_cpu = _to_cpu_uint8(valid_mask.reshape(-1))
    pose_cpu = _to_cpu_float32(_pose_matrix3x4(pose_world_cam).reshape(-1))
    max_points = ((height + stride - 1) // stride) * ((width + stride - 1) // stride)
    out_points = np.full((max_points, 3), np.nan, dtype=np.float32)
    out_count = np.zeros((1,), dtype=np.int32)
    out_bounds = np.zeros((6,), dtype=np.float32)

    kernels = _load_mojo_kernels()
    kernels.unproject_valid_points_f32(
        _tensor_numpy_view(depth_cpu).ctypes.data,
        _tensor_numpy_view(valid_cpu).ctypes.data,
        _tensor_numpy_view(pose_cpu).ctypes.data,
        (
            int(width),
            int(height),
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            int(stride),
        ),
        (
            out_points.ctypes.data,
            out_count.ctypes.data,
            out_bounds.ctypes.data,
        ),
    )
    count = int(out_count[0])
    points_t = torch.from_numpy(out_points[:count]).to(device=device, dtype=torch.float32)
    bounds_t = torch.from_numpy(out_bounds).to(device=device, dtype=torch.float32)
    return points_t, bounds_t


__all__ = [
    "is_mojo_available",
    "is_mojo_thread_context_supported",
    "render_depth_map_mojo",
    "unproject_candidate_points_mojo",
]

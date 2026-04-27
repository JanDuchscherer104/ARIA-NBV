"""Mojo-backed point↔mesh distance helpers for oracle RRI."""

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

from .types import DistanceBreakdown

_MOJO_KERNEL_MODULE = "oracle_distance_kernels"
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
    """Return True when Python-imported Mojo distance kernels may run in this thread."""

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


def _tensor_numpy_view(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.numpy()  # type: ignore[no-any-return]


def _triangles_from_mesh(gt_verts: torch.Tensor, gt_faces: torch.Tensor) -> torch.Tensor:
    verts_cpu = _to_cpu_float32(gt_verts)
    faces_cpu = gt_faces.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return verts_cpu[faces_cpu].contiguous()


def point_mesh_distance_mojo(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    workers: int | None = None,
) -> torch.Tensor:
    """Return per-point distances to the mesh."""

    points_cpu = _to_cpu_float32(points.view(-1, 3))
    triangles_cpu = _to_cpu_float32(triangles.view(-1, 3, 3))
    out = np.empty(points_cpu.shape[0], dtype=np.float32)
    kernels = _load_mojo_kernels()
    kernels.point_mesh_distance_sq_f32(
        _tensor_numpy_view(points_cpu).ctypes.data,
        points_cpu.shape[0],
        _tensor_numpy_view(triangles_cpu).ctypes.data,
        triangles_cpu.shape[0],
        out.ctypes.data,
        _resolve_workers(points_cpu.shape[0], workers),
    )
    np.sqrt(out, out=out)
    return torch.from_numpy(out).to(device=points.device, dtype=points.dtype)


def mesh_point_distance_mojo(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    workers: int | None = None,
) -> torch.Tensor:
    """Return per-triangle distances to the closest point."""

    points_cpu = _to_cpu_float32(points.view(-1, 3))
    triangles_cpu = _to_cpu_float32(triangles.view(-1, 3, 3))
    out = np.empty(triangles_cpu.shape[0], dtype=np.float32)
    kernels = _load_mojo_kernels()
    kernels.triangle_point_distance_sq_f32(
        _tensor_numpy_view(points_cpu).ctypes.data,
        points_cpu.shape[0],
        _tensor_numpy_view(triangles_cpu).ctypes.data,
        triangles_cpu.shape[0],
        out.ctypes.data,
        _resolve_workers(triangles_cpu.shape[0], workers),
    )
    np.sqrt(out, out=out)
    return torch.from_numpy(out).to(device=points.device, dtype=points.dtype)


def chamfer_point_mesh_mojo(
    points: torch.Tensor,
    gt_verts: torch.Tensor,
    gt_faces: torch.Tensor,
    *,
    workers: int | None = None,
) -> DistanceBreakdown:
    """Compute point↔mesh distances with the Mojo kernels."""

    triangles = _triangles_from_mesh(gt_verts, gt_faces)
    point_dist = point_mesh_distance_mojo(points, triangles, workers=workers)
    tri_dist = mesh_point_distance_mojo(points, triangles, workers=workers)
    acc = point_dist.mean() if point_dist.numel() > 0 else torch.zeros((), device=points.device, dtype=points.dtype)
    comp = tri_dist.mean() if tri_dist.numel() > 0 else torch.zeros((), device=points.device, dtype=points.dtype)
    return DistanceBreakdown(accuracy=acc, completeness=comp, bidirectional=acc + comp)


def chamfer_point_mesh_batched_mojo(
    points: torch.Tensor,
    lengths: torch.Tensor,
    gt_verts: torch.Tensor,
    gt_faces: torch.Tensor,
    *,
    workers: int | None = None,
) -> DistanceBreakdown:
    """Compute batched point↔mesh distances with per-candidate loops."""

    if points.ndim != 3:
        raise ValueError(f"Expected batched points of shape (C,P,3); got {tuple(points.shape)}")

    acc_list: list[torch.Tensor] = []
    comp_list: list[torch.Tensor] = []
    bidir_list: list[torch.Tensor] = []
    for idx in range(int(points.shape[0])):
        count = int(lengths[idx].item())
        points_i = points[idx, :count]
        dist_i = chamfer_point_mesh_mojo(points_i, gt_verts, gt_faces, workers=workers)
        acc_list.append(dist_i.accuracy)
        comp_list.append(dist_i.completeness)
        bidir_list.append(dist_i.bidirectional)

    return DistanceBreakdown(
        accuracy=torch.stack(acc_list, dim=0),
        completeness=torch.stack(comp_list, dim=0),
        bidirectional=torch.stack(bidir_list, dim=0),
    )


__all__ = [
    "chamfer_point_mesh_batched_mojo",
    "chamfer_point_mesh_mojo",
    "is_mojo_available",
    "is_mojo_thread_context_supported",
]

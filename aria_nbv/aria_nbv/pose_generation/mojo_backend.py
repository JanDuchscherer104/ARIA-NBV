"""Mojo-backed CPU collision kernels for candidate generation.

This module keeps the existing PyTorch3D and Trimesh implementations untouched
and only activates when ``CollisionBackend.MOJO`` is selected. The Python side
stages contiguous CPU buffers and hands raw addresses to a lazily imported Mojo
extension module.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .types import CandidateContext

_MOJO_KERNEL_MODULE = "mesh_collision_kernels"
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


def _to_cpu_int64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _tensor_numpy_view(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.numpy()  # type: ignore[no-any-return]


def get_mojo_triangles(ctx: "CandidateContext") -> torch.Tensor:
    """Return cached CPU ``(F, 3, 3)`` triangle vertices for the current mesh."""

    cached = ctx.runtime_cache.get("mojo_mesh_triangles_cpu")
    if isinstance(cached, torch.Tensor):
        return cached

    verts_cpu = _to_cpu_float32(ctx.mesh_verts)
    faces_cpu = _to_cpu_int64(ctx.mesh_faces)
    triangles = verts_cpu[faces_cpu].contiguous()
    ctx.runtime_cache["mojo_mesh_triangles_cpu"] = triangles
    return triangles


def point_mesh_distance_mojo(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    workers: int | None = None,
) -> torch.Tensor:
    """Compute exact point-to-triangle distances with Mojo on CPU buffers."""

    points_cpu = _to_cpu_float32(points.view(-1, 3))
    triangles_cpu = _to_cpu_float32(triangles.view(-1, 3, 3))

    points_np = _tensor_numpy_view(points_cpu)
    triangles_np = _tensor_numpy_view(triangles_cpu)
    dist_sq = np.empty(points_cpu.shape[0], dtype=np.float32)

    kernels = _load_mojo_kernels()
    kernels.point_mesh_distance_sq_f32(
        points_np.ctypes.data,
        points_cpu.shape[0],
        triangles_np.ctypes.data,
        triangles_cpu.shape[0],
        dist_sq.ctypes.data,
        _resolve_workers(points_cpu.shape[0], workers),
    )
    np.sqrt(dist_sq, out=dist_sq)
    return torch.from_numpy(dist_sq).to(device=points.device, dtype=points.dtype)


def clearance_mask_mojo(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    min_distance: float,
    workers: int | None = None,
) -> torch.Tensor:
    """Return ``True`` for candidate centers that satisfy the mesh clearance."""

    points_cpu = _to_cpu_float32(points.view(-1, 3))
    triangles_cpu = _to_cpu_float32(triangles.view(-1, 3, 3))

    points_np = _tensor_numpy_view(points_cpu)
    triangles_np = _tensor_numpy_view(triangles_cpu)
    keep = np.empty(points_cpu.shape[0], dtype=np.uint8)

    kernels = _load_mojo_kernels()
    kernels.clearance_mask_f32(
        points_np.ctypes.data,
        points_cpu.shape[0],
        triangles_np.ctypes.data,
        triangles_cpu.shape[0],
        (float(min_distance), _resolve_workers(points_cpu.shape[0], workers)),
        keep.ctypes.data,
    )
    keep_t = torch.from_numpy(keep.astype(np.bool_, copy=False))
    return keep_t.to(device=points.device)


def path_collision_mask_mojo(
    origin: torch.Tensor,
    targets: torch.Tensor,
    triangles: torch.Tensor,
    *,
    workers: int | None = None,
) -> torch.Tensor:
    """Return ``True`` for targets whose line segment from ``origin`` hits the mesh."""

    origin_cpu = _to_cpu_float32(origin.view(3))
    targets_cpu = _to_cpu_float32(targets.view(-1, 3))
    triangles_cpu = _to_cpu_float32(triangles.view(-1, 3, 3))

    origin_np = _tensor_numpy_view(origin_cpu)
    targets_np = _tensor_numpy_view(targets_cpu)
    triangles_np = _tensor_numpy_view(triangles_cpu)
    collide = np.empty(targets_cpu.shape[0], dtype=np.uint8)

    kernels = _load_mojo_kernels()
    kernels.path_collision_mask_f32(
        origin_np.ctypes.data,
        targets_np.ctypes.data,
        targets_cpu.shape[0],
        triangles_np.ctypes.data,
        (triangles_cpu.shape[0], _resolve_workers(targets_cpu.shape[0], workers)),
        collide.ctypes.data,
    )
    collide_t = torch.from_numpy(collide.astype(np.bool_, copy=False))
    return collide_t.to(device=targets.device)


__all__ = [
    "clearance_mask_mojo",
    "get_mojo_triangles",
    "is_mojo_available",
    "path_collision_mask_mojo",
    "point_mesh_distance_mojo",
]

"""Compatibility helpers for optional PyTorch3D usage.

This module centralizes lazy imports so the package can be installed without the
`pytorch3d` extra when callers stay on non-PyTorch3D backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytorch3d.loss.point_mesh_distance import face_point_distance, point_face_distance
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.renderer.cameras import PerspectiveCameras
    from pytorch3d.structures import Meshes
else:
    PerspectiveCameras = Any
    MeshRasterizer = Any
    RasterizationSettings = Any
    Meshes = Any
    point_face_distance = Any
    face_point_distance = Any


def pytorch3d_import_error() -> str | None:
    """Return the PyTorch3D import error, or ``None`` when imports succeed."""

    try:
        import pytorch3d  # noqa: F401
        from pytorch3d import _C  # type: ignore[import-untyped]  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def is_pytorch3d_available() -> bool:
    """Return ``True`` when PyTorch3D imports cleanly."""

    return pytorch3d_import_error() is None


def require_pytorch3d(feature: str = "this code path") -> None:
    """Raise a descriptive error when PyTorch3D is required but unavailable."""

    if is_pytorch3d_available():
        return
    error = pytorch3d_import_error()
    msg = f"PyTorch3D is required for {feature}. Install the `aria-nbv[pytorch3d]` extra."
    if error is not None:
        msg = f"{msg} Import failed with: {error}"
    raise ModuleNotFoundError(msg)


def import_perspective_cameras() -> type[Any]:
    """Import and return ``PerspectiveCameras``."""

    require_pytorch3d("PerspectiveCameras support")
    from pytorch3d.renderer.cameras import PerspectiveCameras as _PerspectiveCameras  # type: ignore[import-untyped]

    return _PerspectiveCameras


def is_perspective_cameras_instance(value: object) -> bool:
    """Return True when ``value`` is a PyTorch3D ``PerspectiveCameras`` batch."""

    try:
        perspective_cameras = import_perspective_cameras()
    except Exception:
        return False
    return isinstance(value, perspective_cameras)


def import_renderer_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Import and return the PyTorch3D renderer classes used by the oracle path."""

    require_pytorch3d("PyTorch3D rendering")
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings  # type: ignore[import-untyped]
    from pytorch3d.structures import Meshes  # type: ignore[import-untyped]

    return MeshRasterizer, RasterizationSettings, Meshes


def import_point_mesh_distance_ops() -> tuple[Any, Any, Any]:
    """Import and return the PyTorch3D point↔mesh distance ops."""

    require_pytorch3d("PyTorch3D point-mesh distance")
    from pytorch3d.loss.point_mesh_distance import (  # type: ignore[import-untyped]
        _DEFAULT_MIN_TRIANGLE_AREA,
        face_point_distance,
        point_face_distance,
    )

    return _DEFAULT_MIN_TRIANGLE_AREA, face_point_distance, point_face_distance


__all__ = [
    "MeshRasterizer",
    "Meshes",
    "PerspectiveCameras",
    "RasterizationSettings",
    "face_point_distance",
    "import_perspective_cameras",
    "import_point_mesh_distance_ops",
    "import_renderer_classes",
    "is_perspective_cameras_instance",
    "is_pytorch3d_available",
    "pytorch3d_import_error",
    "point_face_distance",
    "require_pytorch3d",
]

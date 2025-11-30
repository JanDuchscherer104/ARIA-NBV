"""Geometry helpers for candidate generation and pruning."""

from __future__ import annotations

import torch

DEVICE_FWD = [0.0, 0.0, 1.0]


def point_mesh_distance(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute point-to-mesh distances using PyTorch3D.

    Args:
        points: ``(N, 3)`` points in world frame.
        verts: ``(V, 3)`` mesh vertices.
        faces: ``(F, 3)`` mesh faces (indices into ``verts``).

    Returns:
        ``(N,)`` distances in metres on the same device/dtype as ``points``.
    """

    from pytorch3d.loss.point_mesh_distance import (  # type: ignore[import-untyped]
        _DEFAULT_MIN_TRIANGLE_AREA,
        point_face_distance,
    )

    device = points.device
    points = points.to(device)
    verts = verts.to(device)
    faces = faces.to(device)

    tris = verts[faces]
    points_first_idx = torch.zeros(1, device=device, dtype=torch.int64)
    tris_first_idx = torch.zeros(1, device=device, dtype=torch.int64)
    max_points = points.shape[0]

    dist_sq = point_face_distance(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        _DEFAULT_MIN_TRIANGLE_AREA,
    )
    return torch.sqrt(dist_sq)


__all__ = ["point_mesh_distance"]

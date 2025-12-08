"""Low-level RRI metric primitives (point↔mesh distances and Chamfer variants).

This module wraps PyTorch3D geometric distance implementations so downstream
code can compute accuracy, completeness, and bidirectional Chamfer distances in
a uniform, torch-first manner. Only the GPU-accelerated PyTorch3D path is
supported; CPU fallbacks were removed to keep the hot path simple and
consistent. The functions remain thin: callers must supply pre-sampled point
clouds and mesh tensors. Directional components are returned separately to
expose the accuracy/completeness split described in ``surface_metrics.qmd``.
"""

from __future__ import annotations

import torch
from pytorch3d.loss import chamfer_distance  # type: ignore[import-untyped]
from pytorch3d.loss.point_mesh_distance import (  # type: ignore[import-untyped]
    _DEFAULT_MIN_TRIANGLE_AREA,
    face_point_distance,
    point_face_distance,
)
from torch import Tensor

from .types import DistanceAggregation, DistanceBreakdown


def chamfer_point_mesh(
    points: Tensor,
    gt_verts: Tensor,
    gt_faces: Tensor,
) -> DistanceBreakdown:
    """Compute accuracy, completeness, and bidirectional Chamfer for P<->M."""

    lengths = torch.tensor([points.shape[0]], device=points.device, dtype=torch.long)
    padded = points.unsqueeze(0)

    dist = chamfer_point_mesh_batched(padded, lengths, gt_verts, gt_faces)
    return DistanceBreakdown(
        accuracy=dist.accuracy.squeeze(0),
        completeness=dist.completeness.squeeze(0),
        bidirectional=dist.bidirectional.squeeze(0),
    )


def chamfer_point_mesh_batched(
    points: Tensor,
    lengths: Tensor,
    gt_verts: Tensor,
    gt_faces: Tensor,
) -> DistanceBreakdown:
    """Chamfer-like point↔mesh distance per example (fully vectorised).

    Returns per-candidate accuracy (P→M), completeness (M→P), and bidirectional
    sums without Python-level loops.
    """

    if points.ndim != 3:
        raise ValueError(f"Expected batched points of shape (C,P,3); got {tuple(points.shape)}")

    bsz, max_p, _ = points.shape
    lengths = lengths.clamp(max=max_p)

    mask = torch.arange(max_p, device=points.device).unsqueeze(0) < lengths.unsqueeze(1)
    points_packed = points[mask]  # (Ptot, 3)

    points_first_idx = torch.zeros(bsz, device=points.device, dtype=torch.int64)
    points_first_idx[1:] = lengths.cumsum(0)[:-1]
    max_points = int(lengths.max().item())
    point_to_cloud_idx = torch.repeat_interleave(torch.arange(bsz, device=points.device), lengths)

    v = gt_verts.shape[0]
    f = gt_faces.shape[0]
    verts_packed = gt_verts.repeat(bsz, 1)
    face_offsets = torch.arange(bsz, device=gt_faces.device, dtype=gt_faces.dtype).unsqueeze(1) * v
    faces_packed = (gt_faces.unsqueeze(0) + face_offsets.unsqueeze(-1)).reshape(-1, 3)
    tris = verts_packed[faces_packed]
    tris_first_idx = torch.arange(0, bsz * f, f, device=points.device, dtype=torch.int64)
    max_tris = f
    tri_to_mesh_idx = torch.repeat_interleave(torch.arange(bsz, device=points.device), f)
    num_tris_per_mesh = torch.full((bsz,), f, device=points.device, dtype=points.dtype)

    point_to_face = point_face_distance(
        points_packed, points_first_idx, tris, tris_first_idx, max_points, _DEFAULT_MIN_TRIANGLE_AREA
    )
    num_points_per_cloud = lengths.to(points.dtype).clamp(min=1)
    weights_p = 1.0 / num_points_per_cloud.gather(0, point_to_cloud_idx).float()
    acc = torch.zeros(bsz, device=points.device, dtype=points.dtype)
    acc.scatter_add_(0, point_to_cloud_idx, point_to_face * weights_p)

    face_to_point = face_point_distance(
        points_packed, points_first_idx, tris, tris_first_idx, max_tris, _DEFAULT_MIN_TRIANGLE_AREA
    )
    weights_t = 1.0 / num_tris_per_mesh.gather(0, tri_to_mesh_idx).float()
    comp = torch.zeros(bsz, device=points.device, dtype=points.dtype)
    comp.scatter_add_(0, tri_to_mesh_idx, face_to_point * weights_t)

    return DistanceBreakdown(
        accuracy=acc,
        completeness=comp,
        bidirectional=acc + comp,
    )


# def chamfer_point_mesh(
#     points: Tensor,  # (N, 3)
#     gt_verts: Tensor,  # (V, 3)
#     gt_faces: Tensor,  # (F, 3) int64
#     *,
#     num_gt_samples: int = 20000,
#     eps: float = 1e-12,
# ) -> DistanceBreakdown:
#     """Compute accuracy, completeness, and bidirectional Chamfer for P ↔ M.

#     Conceptual definition (see ``docs/contents/theory/surface_metrics.qmd``):

#     - Accuracy: :math:`d_{P\\to M} = \\frac{1}{|P|} \\sum_{p\\in P} \\min_{x\\in M} \\|p-x\\|_2`
#     - Completeness: :math:`d_{M\\to P} = \\frac{1}{|M|} \\sum_{x\\in M} \\min_{p\\in P} \\|x-p\\|_2`
#     - Bidirectional: :math:`d_{P\\leftrightarrow M} = d_{P\\to M} + d_{M\\to P}`

#     Args:
#         points: ``Tensor["N", 3]`` prediction point cloud in world frame.
#         gt_verts: ``Tensor["V", 3]`` ground-truth mesh vertices.
#         gt_faces: ``Tensor["F", 3]`` ground-truth triangular faces (int64).

#     Returns:
#         DistanceBreakdown with three tensors

#     Note:
#         Uses ``point_mesh_face_distance`` which internally computes both
#         point2face and face2point terms.
#     """
#     pcl = Pointclouds(points.unsqueeze(0))
#     mesh = Meshes(verts=[gt_verts], faces=[gt_faces])

#     # -------- Accuracy: P -> M (exact point-to-triangle distance) --------
#     pts = pcl.points_packed()  # (P, 3)
#     pts_first = pcl.cloud_to_packed_first_idx()  # (1,)
#     max_pts = int(pcl.num_points_per_cloud().max())

#     verts_packed = mesh.verts_packed()
#     faces_packed = mesh.faces_packed()
#     tris = verts_packed[faces_packed]  # (T, 3, 3)
#     tris_first = mesh.mesh_to_faces_packed_first_idx()  # (1,)

#     d2_p2m = point_face_distance(pts, pts_first, tris, tris_first, max_pts)  # (P,) squared
#     if squared:
#         acc = d2_p2m.mean()
#     else:
#         acc = (d2_p2m + eps).sqrt().mean()

#     # -------- Completeness: M -> P (area-uniform GT surface samples) -----
#     q = sample_points_from_meshes(mesh, num_samples=num_gt_samples)  # (1, S, 3)
#     knn = knn_points(q, points.unsqueeze(0), K=1)  # squared dists
#     d2_m2p = knn.dists[..., 0]  # (1, S)
#     if squared:
#         comp = d2_m2p.mean()
#     else:
#         comp = (d2_m2p + eps).sqrt().mean()

#     bidir = acc + comp
#     return DistanceBreakdown(accuracy=acc, completeness=comp, bidirectional=bidir)


def chamfer_point_point(
    points_a: Tensor,
    points_b: Tensor,
    *,
    reduction: DistanceAggregation = DistanceAggregation.MEAN,
) -> DistanceBreakdown:
    """Chamfer distance between two point clouds (fallback when no mesh).

    Uses ``pytorch3d.loss.chamfer_distance`` to obtain both directional terms.
    This is primarily a utility for ablations or validation when only point
    clouds are available (e.g., comparing fused reconstructions without GT).
    """

    if points_a.ndim != 2 or points_b.ndim != 2:
        raise ValueError("Expected unbatched point sets of shape (N,3) and (M,3)")

    a = points_a.unsqueeze(0)
    b = points_b.unsqueeze(0)

    point_reduction: str | None
    batch_reduction: str | None
    if reduction == DistanceAggregation.NONE:
        point_reduction = None  # return per-point tensors
        batch_reduction = None
    else:
        point_reduction = reduction.value  # "mean" | "sum"
        batch_reduction = None  # keep per-batch scalar

    loss_ab, _ = chamfer_distance(
        a,
        b,
        batch_reduction=batch_reduction,
        point_reduction=point_reduction,
    )

    if isinstance(loss_ab, tuple):
        # point_reduction=None → forward/backward per-point distances
        forward, backward = loss_ab
        acc = forward.squeeze(0)
        comp = backward.squeeze(0)
    else:
        # symmetric scalar per batch
        acc = comp = loss_ab.squeeze()

    bidir = acc + comp
    return DistanceBreakdown(accuracy=acc, completeness=comp, bidirectional=bidir)


__all__ = [
    "chamfer_point_mesh",
    "chamfer_point_mesh_batched",
    "chamfer_point_point",
    "DistanceBreakdown",
]

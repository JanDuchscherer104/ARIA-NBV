"""Vectorised depth→point-cloud conversion for candidate renders.

Consumes a single :class:`EfmSnippetView` and matching :class:`CandidateDepths`
to produce padded per-candidate point clouds, fused clouds with the collapsed
semi-dense SLAM reconstruction, and a combined occupancy extent for cropping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)

from ..data_handling import EfmSnippetView
from .candidate_depth_renderer import CandidateDepths

Tensor = torch.Tensor


@dataclass(slots=True)
class CandidatePointClouds:
    """Batched candidate point clouds plus fused semi-dense reconstruction."""

    points: Tensor
    """Tensor['B', 'P', 3] padded candidate point clouds (world frame)."""
    lengths: Tensor
    """Tensor['B'] actual point counts per candidate."""
    semidense_points: Tensor
    """Tensor['K', 3] collapsed semi-dense SLAM point cloud."""
    semidense_length: Tensor
    """Tensor[1] number of valid semi-dense points."""
    occupancy_bounds: Tensor
    """Tensor[6] = [xmin, xmax, ymin, ymax, zmin, zmax] covering snippet + candidates."""


def build_candidate_pointclouds(
    sample: EfmSnippetView,
    batch: CandidateDepths,
    *,
    stride: int = 1,
) -> CandidatePointClouds:
    """Convert stacked depth maps into batched point clouds and fuse with SLAM."""
    depths = batch.depths
    cameras = batch.p3d_cameras

    if depths.ndim != 3:
        raise ValueError(f"Expected depths of shape (B,H,W), got {tuple(depths.shape)}")

    padded, lengths = _backproject_depths_p3d_batch(
        depths=depths,
        mask_valid=batch.depths_valid_mask,
        cameras=cameras,
        stride=stride,
    )

    device, dtype = padded.device, padded.dtype

    semidense_pts = sample.semidense.collapse_points()
    semidense_pts_t = torch.as_tensor(semidense_pts, device=device, dtype=dtype)
    semidense_len = torch.tensor([semidense_pts_t.shape[0]], device=device, dtype=torch.long)

    occupancy_bounds = _compute_bounds(sample.get_occupancy_extend(), padded, lengths, semidense_pts_t)

    return CandidatePointClouds(
        points=padded,
        lengths=lengths,
        semidense_points=semidense_pts_t,
        semidense_length=semidense_len,
        occupancy_bounds=occupancy_bounds,
    )


def _backproject_depths_p3d_batch(
    depths: Tensor,
    mask_valid: Tensor,
    cameras: PerspectiveCameras,
    *,
    stride: int = 1,
) -> tuple[Tensor, Tensor]:
    """Vectorised backprojection of stacked depth maps via PyTorch3D.

    Args:
        depths: Tensor['B', 'H', 'W'] depth maps.
        mask_valid: Tensor['B', 'H', 'W'] boolean valid pixel masks.
        cameras: PerspectiveCameras for each depth map.
        stride: Subsampling stride for pixels.

    Returns:
        padded: Tensor['B', 'Pmax', 3] world-frame points.
        lengths: Tensor['B'] valid point counts per candidate.
    """
    bsz, h, w = depths.shape
    yy = torch.arange(0, h, stride, device=depths.device)
    xx = torch.arange(0, w, stride, device=depths.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    p = gy.numel()

    depth_sub = depths[:, gy, gx].reshape(bsz, p)  # (B, P)
    mask = torch.isfinite(depth_sub) & mask_valid[:, gy, gx].reshape(bsz, p)

    depth_filtered = torch.where(mask, depth_sub, torch.zeros_like(depth_sub))

    # Convert pixel centers to PyTorch3D NDC coordinates (+X left, +Y up) and
    # unproject in that space. This matches the convention used by the
    # PyTorch3D rasterizer for ``in_ndc=False`` cameras with non-square images.
    #
    # Using pixel coordinates directly (``from_ndc=False``) produces points in a
    # different screen convention and does *not* match the rasterizer, which
    # leads to backprojected points that do not lie on the rendered mesh.
    gx_flat = gx.reshape(-1).to(depths.dtype) + 0.5
    gy_flat = gy.reshape(-1).to(depths.dtype) + 0.5
    scale = float(min(h, w))
    x_ndc = -(gx_flat - (w * 0.5)) * (2.0 / scale)
    y_ndc = -(gy_flat - (h * 0.5)) * (2.0 / scale)

    x_ndc = x_ndc.unsqueeze(0).expand(bsz, -1)
    y_ndc = y_ndc.unsqueeze(0).expand(bsz, -1)

    xy_depth = torch.stack([x_ndc, y_ndc, depth_filtered.to(depths.dtype)], dim=-1)  # (B, P, 3)
    pts_world = cameras.unproject_points(xy_depth, world_coordinates=True, from_ndc=True)  # (B, P, 3)

    lengths = mask.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    if max_len == 0:
        return torch.empty(bsz, 0, 3, device=depths.device, dtype=depths.dtype), lengths

    # Compact valid points per batch without sorting; use running counts.
    padded = torch.full((bsz, max_len, 3), torch.nan, device=depths.device, dtype=depths.dtype)
    cumsum = mask.cumsum(dim=1) - 1  # positions of each valid point within its batch
    batch_idx, flat_idx = torch.nonzero(mask, as_tuple=True)
    pos_idx = cumsum[batch_idx, flat_idx]
    padded[batch_idx, pos_idx] = pts_world[batch_idx, flat_idx]

    return padded, lengths


# def _backproject_depths_p3d_batch(
#     depths: Tensor,
#     mask_valid: Tensor,
#     cameras: PerspectiveCameras,
#     *,
#     stride: int = 1,
# ) -> tuple[Tensor, Tensor]:
#     """Vectorised backprojection of stacked depth maps via PyTorch3D.
#     Args:
#         depths: Tensor['B', 'H', 'W'] depth maps.
#         mask_valid: Tensor['B', 'H', 'W'] boolean valid pixel masks.
#         cameras: PerspectiveCameras for each depth map.
#         stride: Subsampling stride for pixels.

#     Returns:
#         padded: Tensor['B', 'Pmax', 3] world-frame points.
#         lengths: Tensor['B'] valid point counts per candidate.
#     """

#     bsz, h, w = depths.shape
#     yy = torch.arange(0, h, stride, device=depths.device)
#     xx = torch.arange(0, w, stride, device=depths.device)
#     gy, gx = torch.meshgrid(yy, xx, indexing="ij")
#     p = gy.numel()

#     depth_sub = depths[:, gy, gx].reshape(bsz, p)  # (B, P)
#     mask = torch.isfinite(depth_sub) & mask_valid[:, gy, gx].reshape(bsz, p)

#     depth_filtered = torch.where(mask, depth_sub, torch.zeros_like(depth_sub))

#     u = gx.reshape(-1).to(depths.dtype) + 0.5
#     v = gy.reshape(-1).to(depths.dtype) + 0.5

#     # convert to NDC (PyTorch3D convention: +X left, +Y up)
#     s = float(min(h, w))
#     x_ndc = -(u - (w * 0.5)) * (2.0 / s)
#     y_ndc = -(v - (h * 0.5)) * (2.0 / s)

#     x_ndc = x_ndc.unsqueeze(0).expand(bsz, -1)
#     y_ndc = y_ndc.unsqueeze(0).expand(bsz, -1)

#     # build xy_depth in NDC
#     xy_depth = torch.stack([x_ndc, y_ndc, depth_filtered], dim=-1)  # (B,P,3

#     pts_world = cameras.unproject_points(xy_depth, world_coordinates=True, from_ndc=True)  # (B,P,3)

#     lengths = mask.sum(dim=1)
#     max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
#     if max_len == 0:
#         return torch.empty(bsz, 0, 3, device=depths.device, dtype=depths.dtype), lengths

#     # Compact valid points per batch without sorting; use running counts.
#     padded = torch.full((bsz, max_len, 3), torch.nan, device=depths.device, dtype=depths.dtype)
#     cumsum = mask.cumsum(dim=1) - 1  # positions of each valid point within its batch
#     batch_idx, flat_idx = torch.nonzero(mask, as_tuple=True)
#     pos_idx = cumsum[batch_idx, flat_idx]
#     padded[batch_idx, pos_idx] = pts_world[batch_idx, flat_idx]

#     return padded, lengths


def _compute_bounds(
    snippet_bounds: Tensor,
    padded: Tensor,
    lengths: Tensor,
    semidense: Tensor,
) -> Tensor:
    """Combine snippet occupancy bounds with candidate and semi-dense extents."""
    out = snippet_bounds.to(device=padded.device, dtype=padded.dtype)
    x_min, x_max, y_min, y_max, z_min, z_max = out.unbind()

    if padded.numel() > 0 and padded.shape[1] > 0:
        mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        pts = padded[mask]
        if pts.numel() > 0:
            pmin = torch.amin(pts, dim=0)
            pmax = torch.amax(pts, dim=0)
            x_min, x_max = torch.minimum(x_min, pmin[0]), torch.maximum(x_max, pmax[0])
            y_min, y_max = torch.minimum(y_min, pmin[1]), torch.maximum(y_max, pmax[1])
            z_min, z_max = torch.minimum(z_min, pmin[2]), torch.maximum(z_max, pmax[2])

    if semidense.numel() > 0:
        smin = torch.amin(semidense, dim=0)
        smax = torch.amax(semidense, dim=0)
        x_min, x_max = torch.minimum(x_min, smin[0]), torch.maximum(x_max, smax[0])
        y_min, y_max = torch.minimum(y_min, smin[1]), torch.maximum(y_max, smax[1])
        z_min, z_max = torch.minimum(z_min, smin[2]), torch.maximum(z_max, smax[2])

    return torch.stack([x_min, x_max, y_min, y_max, z_min, z_max], dim=0)


__all__ = ["CandidatePointClouds", "build_candidate_pointclouds"]

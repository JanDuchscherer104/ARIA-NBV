r"""Utilities to back-project rendered depth maps into world-frame point clouds.

This module centralises depth unprojection for candidate renders to avoid frame
confusion between the PyTorch3D renderer (which outputs metric ``z`` depth in
the physical camera frame) and downstream visualisations or fusion steps.

All functions assume:
    * ``depth`` is metric depth along the camera +Z axis (same convention as
      `pytorch3d.renderer.MeshRasterizer` with ``in_ndc=False``).
    * ``pose_world_cam`` is a `efm3d.aria.pose.PoseTW` storing
      **world ← camera** extrinsics (LUF camera frame).
    * ``camera`` is the matching `efm3d.aria.camera.CameraTW` carrying
      intrinsics (and, if batched, per-candidate extrinsics that align with the
      provided depths/poses).

The returned points live in the same VIO/world frame as the ASE mesh and
semidense history. Conceptually a valid depth pixel $d(u,v)$ maps to
$\mathbf{p}_w = T_{w \leftarrow c}\,\pi^{-1}(u,v,d)$; crop and RRI code then
decide whether the point contributes to scene-level or target-level scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .candidate_depth_renderer import CandidateDepths


def backproject_depths_p3d_batch(
    depths: torch.Tensor,
    mask_valid: torch.Tensor,
    cameras: PerspectiveCameras,
    *,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project a batch of PyTorch3D depth maps to world-frame points.

    Args:
        depths: ``Tensor["B", "H", "W"]`` metric z-depth maps in metres.
        mask_valid: ``Tensor["B", "H", "W"]`` boolean masks for usable pixels.
        cameras: One `pytorch3d.renderer.PerspectiveCameras` entry per
            depth map, carrying world-from-camera extrinsics.
        stride: Pixel subsampling stride.

    Returns:
        Pair ``(padded, lengths)`` where ``padded`` is
        ``Tensor["B", "Pmax", 3]`` in world coordinates and ``lengths`` is
        ``Tensor["B"]`` with the valid count per candidate.
    """
    if depths.ndim != 3:
        raise ValueError(f"Expected depths of shape (B,H,W), got {tuple(depths.shape)}")
    if mask_valid.shape != depths.shape:
        raise ValueError(f"mask_valid shape {tuple(mask_valid.shape)} must match depths {tuple(depths.shape)}")
    if stride < 1:
        raise ValueError(f"stride must be >=1, got {stride}")

    bsz, height, width = depths.shape
    yy = torch.arange(0, height, stride, device=depths.device)
    xx = torch.arange(0, width, stride, device=depths.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    num_pixels = gy.numel()

    depth_sub = depths[:, gy, gx].reshape(bsz, num_pixels)
    mask = torch.isfinite(depth_sub) & mask_valid[:, gy, gx].reshape(bsz, num_pixels)
    depth_filtered = torch.where(mask, depth_sub, torch.zeros_like(depth_sub))

    gx_flat = gx.reshape(-1).to(depths.dtype) + 0.5
    gy_flat = gy.reshape(-1).to(depths.dtype) + 0.5
    scale = float(min(height, width))
    x_ndc = -(gx_flat - (width * 0.5)) * (2.0 / scale)
    y_ndc = -(gy_flat - (height * 0.5)) * (2.0 / scale)

    xy_depth = torch.stack(
        [
            x_ndc.unsqueeze(0).expand(bsz, -1),
            y_ndc.unsqueeze(0).expand(bsz, -1),
            depth_filtered.to(depths.dtype),
        ],
        dim=-1,
    )
    pts_world = cameras.unproject_points(xy_depth, world_coordinates=True, from_ndc=True)

    lengths = mask.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    if max_len == 0:
        return torch.empty(bsz, 0, 3, device=depths.device, dtype=depths.dtype), lengths

    padded = torch.full((bsz, max_len, 3), torch.nan, device=depths.device, dtype=depths.dtype)
    cumsum = mask.cumsum(dim=1) - 1
    batch_idx, flat_idx = torch.nonzero(mask, as_tuple=True)
    padded[batch_idx, cumsum[batch_idx, flat_idx]] = pts_world[batch_idx, flat_idx]

    return padded, lengths


def backproject_depth_with_p3d(
    depth: torch.Tensor,
    cameras: PerspectiveCameras,
    valid_mask: torch.Tensor,
    *,
    stride: int = 1,
    max_points: int | None = None,
) -> torch.Tensor:
    """Back-project a single depth map using PyTorch3D cameras and a validity mask.

    Args:
        depth: ``Tensor["H", "W"]`` metric depth (metres) in the camera frame.
        cameras: Matching `pytorch3d.renderer.PerspectiveCameras`.
        valid_mask: ``Tensor["H", "W"]`` boolean mask for usable pixels
            (e.g. zclose/zfar filtering from `CandidateDepths`).
        stride: Optional subsampling stride in pixel space.
        max_points: Optional cap on returned points (random subset when exceeded).

    Returns:
        ``Tensor["N", 3"]`` of world-frame points for valid pixels only.
    """
    # depth: (H, W) metric z from MeshRasterizer (in_ndc=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected (H,W) depth, got {tuple(depth.shape)}")
    if valid_mask.shape != depth.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} must match depth {tuple(depth.shape)}")
    if stride < 1:
        raise ValueError(f"stride must be >=1, got {stride}")

    padded, lengths = backproject_depths_p3d_batch(
        depths=depth.unsqueeze(0),
        mask_valid=valid_mask.unsqueeze(0),
        cameras=cameras,
        stride=stride,
    )
    pts_world = padded[0, : int(lengths[0].item())]

    if max_points is not None and pts_world.shape[0] > max_points:
        idx = torch.randperm(pts_world.shape[0], device=pts_world.device)[:max_points]
        pts_world = pts_world[idx]

    return pts_world


def backproject_batch(
    batch: CandidateDepths,
    *,
    stride: int = 1,
    zfar: float | None = None,
    max_points: int | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Back-project a `CandidateDepths` batch into a merged world point cloud.

    Args:
        batch: Rendered candidate depths + poses.
        stride: Pixel subsampling stride applied to each depth map.
        zfar: Optional upper depth bound; defaults to per-map max.
        max_points: Optional cap on total points (post-concat).
        candidate_indices: Optional subset (local indices in the batch).

    Returns:
        ``Tensor["N", 3"]`` concatenated world-frame points.
    """

    depths = batch.depths
    valid_masks = batch.depths_valid_mask
    cameras = batch.p3d_cameras

    if depths.ndim != 3:
        raise ValueError(f"Expected (B,H,W) depths, got {tuple(depths.shape)}")
    if cameras is None:
        raise ValueError("CandidateDepths.p3d_cameras is required for backprojection.")

    num = depths.shape[0]
    use_idxs = list(candidate_indices) if candidate_indices is not None else list(range(num))

    pts_all: list[torch.Tensor] = []
    for i in use_idxs:
        depth_i = depths[i]
        valid_i = valid_masks[i]
        if zfar is not None:
            valid_i = valid_i & (depth_i < float(zfar))

        pts_world = backproject_depth_with_p3d(
            depth=depth_i,
            cameras=cameras[i],
            valid_mask=valid_i,
            stride=stride,
            max_points=max_points,
        )
        if pts_world.numel() > 0:
            pts_all.append(pts_world)

    if not pts_all:
        return torch.empty(0, 3, device=depths.device, dtype=depths.dtype)

    pts_concat = torch.cat(pts_all, dim=0)
    if max_points is not None and pts_concat.shape[0] > max_points:
        idx = torch.randperm(pts_concat.shape[0], device=pts_concat.device)[:max_points]
        pts_concat = pts_concat[idx]

    return pts_concat


__all__ = ["backproject_depth_with_p3d", "backproject_depths_p3d_batch", "backproject_batch"]

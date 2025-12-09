"""Utilities to back-project rendered depth maps into world-frame point clouds.

This module centralises depth unprojection for candidate renders to avoid frame
confusion between the PyTorch3D renderer (which outputs metric ``z`` depth in
the physical camera frame) and downstream visualisations or fusion steps.

All functions assume:
    * ``depth`` is metric depth along the camera +Z axis (same convention as
      :class:`~pytorch3d.renderer.MeshRasterizer` with ``in_ndc=False``).
    * ``pose_world_cam`` is a :class:`efm3d.aria.pose.PoseTW` storing
      **world ← camera** extrinsics (LUF camera frame).
    * ``camera`` is the matching :class:`efm3d.aria.camera.CameraTW` carrying
      intrinsics (and, if batched, per-candidate extrinsics that align with the
      provided depths/poses).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from .candidate_depth_renderer import CandidateDepths

# def backproject_depth(
#     depth: torch.Tensor,
#     pose_world_cam: PoseTW,
#     camera: PerspectiveCameras,
#     *,
#     stride: int = 1,
#     zfar: float | None = None,
#     max_points: int | None = None,
# ) -> torch.Tensor:
#     """Back-project a single depth map into world-frame 3D points.

#     Args:
#         depth: ``Tensor["H", "W"]`` metric z-depth in metres.
#         pose_world_cam: ``PoseTW`` (world ← cam) matching ``depth``.
#         camera: ``CameraTW`` (single entry) whose intrinsics were used for
#             rendering this depth map.
#         stride: Optional subsampling stride in pixel space to thin points.
#         zfar: Optional max depth; values >= ``zfar`` are discarded.
#         max_points: Optional cap on returned points (random subset).

#     Returns:
#         ``Tensor["N", 3"]`` of 3D points in world coordinates. ``N`` may be
#         zero if no valid hits remain.
#     """

#     if depth.ndim != 2:
#         raise ValueError(f"Expected (H,W) depth, got {tuple(depth.shape)}")
#     if stride < 1:
#         raise ValueError(f"stride must be >=1, got {stride}")

#     h, w = depth.shape
#     yy = torch.arange(0, h, stride, device=depth.device, dtype=depth.dtype)
#     xx = torch.arange(0, w, stride, device=depth.device, dtype=depth.dtype)
#     gy, gx = torch.meshgrid(yy, xx, indexing="ij")

#     depth_samples = depth[gy.long(), gx.long()].reshape(-1)
#     coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

#     finite_mask = torch.isfinite(depth_samples)
#     if zfar is not None:
#         finite_mask &= depth_samples < float(zfar) * 0.99
#     depth_samples = depth_samples[finite_mask]
#     coords = coords[finite_mask]

#     if depth_samples.numel() == 0:
#         return torch.empty(0, 3, device=depth.device, dtype=depth.dtype)

#     # Use pinhole intrinsics (matching PyTorch3D render path; distortion ignored).
#     cam_single = camera if camera.tensor().ndim == 1 else camera[0]
#     fx, fy, cx, cy = _scalar_intrinsics(cam_single)
#     print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
#     fx_t = torch.tensor(fx, device=depth.device, dtype=depth.dtype)
#     fy_t = torch.tensor(fy, device=depth.device, dtype=depth.dtype)
#     cx_t = torch.tensor(cx, device=depth.device, dtype=depth.dtype)
#     cy_t = torch.tensor(cy, device=depth.device, dtype=depth.dtype)

#     x_cam = -(coords[:, 0] - cx_t + 0.5) / fx_t * depth_samples
#     y_cam = -(coords[:, 1] - cy_t + 0.5) / fy_t * depth_samples
#     z_cam = depth_samples
#     pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

#     pts_world = pose_world_cam.transform(pts_cam)

#     return pts_world


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
        cameras: Matching :class:`~pytorch3d.renderer.PerspectiveCameras`.
        valid_mask: ``Tensor["H", "W"]`` boolean mask for usable pixels
            (e.g. zclose/zfar filtering from :class:`CandidateDepths`).
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

    h, w = depth.shape

    yy = torch.arange(0, h, stride, device=depth.device)
    xx = torch.arange(0, w, stride, device=depth.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")

    depth_sub = depth[gy.long(), gx.long()]
    mask = torch.isfinite(depth_sub) & valid_mask[gy.long(), gx.long()]
    # if zfar is not None:
    #     mask &= depth_sub < float(zfar) * 0.99
    # if zclose is not None:
    #     mask &= depth_sub > float(zclose)

    flat_mask = mask.reshape(-1)
    if not flat_mask.any():
        return torch.empty(0, 3, device=depth.device, dtype=depth.dtype)

    depth_flat = depth_sub.reshape(-1)
    coords_x = gx.reshape(-1)[flat_mask]
    coords_y = gy.reshape(-1)[flat_mask]
    depth_samples = depth_flat[flat_mask]

    coords_x = (w - 1) - coords_x
    coords_y = (h - 1) - coords_y

    pixels = torch.stack([coords_x, coords_y, depth_samples], dim=1)  # (N,3)
    pts_world = cameras.unproject_points(pixels, world_coordinates=True, from_ndc=False)

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
    """Back-project a :class:`CandidateDepths` batch into a merged world point cloud.

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


__all__ = ["backproject_depth_with_p3d", "backproject_batch"]

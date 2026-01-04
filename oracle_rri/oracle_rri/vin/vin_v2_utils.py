"""VIN v2 helper dataclasses and utility functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from efm3d.aria.pose import PoseTW
from torch import Tensor

from ..data.efm_views import EfmSnippetView, VinSnippetView

if TYPE_CHECKING:
    from efm3d.aria.pose import PoseTW as PoseTWT


@dataclass(slots=True)
class PreparedInputs:
    """Prepared inputs for VIN v2 forward pass."""

    pose_world_cam: PoseTW
    """``PoseTW["B N 12"]`` Candidate poses in world frame."""

    pose_world_rig_ref: PoseTW
    """``PoseTW["B 12"]`` Reference rig pose in world frame."""

    t_world_voxel: PoseTW
    """``PoseTW["B 12"]`` World←voxel pose for the EVL voxel grid."""

    batch_size: int
    """Batch size inferred from candidate poses."""

    num_candidates: int
    """Number of candidates per batch."""

    device: torch.device
    """Device for tensors in the forward pass."""

    snippet: EfmSnippetView | VinSnippetView | None
    """Optional snippet view for semidense features."""


@dataclass(slots=True)
class PoseFeatures:
    """Pose-related features for VIN v2."""

    pose_enc: Tensor
    """``Tensor["B N E", float32]`` Pose encoder output."""

    pose_vec: Tensor
    """``Tensor["B N D", float32]`` Pose vector fed into the pose encoder."""

    candidate_center_rig_m: Tensor
    """``Tensor["B N 3", float32]`` Candidate centers in reference rig frame."""


@dataclass(slots=True)
class FieldBundle:
    """Scene field tensors for VIN v2."""

    field_in: Tensor
    """``Tensor["B C_in D H W", float32]`` Raw scene field."""

    field: Tensor
    """``Tensor["B C_out D H W", float32]`` Projected scene field."""

    aux: dict[str, Tensor]
    """Auxiliary channels (e.g. counts_norm, occ_pr)."""


@dataclass(slots=True)
class GlobalContext:
    """Global context features computed from the scene field."""

    pos_grid: Tensor
    """``Tensor["B 3 D H W", float32]`` Normalized position grid."""

    global_feat: Tensor
    """``Tensor["B N C", float32]`` Pose-conditioned global features."""


def ensure_candidate_batch(candidate_poses_world_cam: PoseTWT) -> PoseTWT:
    """Ensure candidate poses are batched as ``(B,N,12)``."""
    if candidate_poses_world_cam.ndim == 2:  # N x 12
        return candidate_poses_world_cam.unsqueeze(0)
    if candidate_poses_world_cam.ndim != 3:
        raise ValueError(
            "candidate_poses_world_cam must have shape (N,12) or (B,N,12).",
        )
    return candidate_poses_world_cam


def ensure_pose_batch(pose: PoseTWT, *, batch_size: int, name: str) -> PoseTWT:
    """Broadcast a pose to ``(B,12)`` to match the candidate batch size."""
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    elif pose.ndim != 2:
        raise ValueError(
            f"{name} must have shape (12,) or (B,12), got ndim={pose.ndim}.",
        )

    return pose


def pos_grid_from_pts_world(
    pts_world: Tensor,
    *,
    t_world_voxel: PoseTW,
    pose_world_rig_ref: PoseTW,
    voxel_extent: Tensor,
    grid_shape: tuple[int, int, int],
) -> Tensor:
    """Convert voxel center points to a per-axis normalized position grid in the reference rig frame.

    If the provided points correspond to a larger grid (e.g. before a valid-kernel
    Conv3d shrink), the grid is center-cropped to ``grid_shape``.
    """

    def _infer_pts_shape(num_pts: int, target_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        d_t, h_t, w_t = target_shape
        if num_pts == d_t * h_t * w_t:
            return target_shape
        for pad in range(1, 4):
            d_p, h_p, w_p = d_t + 2 * pad, h_t + 2 * pad, w_t + 2 * pad
            if num_pts == d_p * h_p * w_p:
                return (d_p, h_p, w_p)
        raise ValueError(
            "pts_world size mismatch: "
            f"got {num_pts} points; expected {d_t * h_t * w_t} "
            f"for grid_shape {target_shape} or a symmetric padding variant.",
        )

    def _center_crop(
        grid: Tensor,
        target_shape: tuple[int, int, int],
    ) -> Tensor:
        d0, h0, w0 = int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
        d_t, h_t, w_t = target_shape
        if (d0, h0, w0) == target_shape:
            return grid
        if d0 < d_t or h0 < h_t or w0 < w_t:
            raise ValueError(
                f"pts_world grid {d0, h0, w0} smaller than target {target_shape}.",
            )
        if (d0 - d_t) % 2 != 0 or (h0 - h_t) % 2 != 0 or (w0 - w_t) % 2 != 0:
            raise ValueError(
                f"pts_world grid {d0, h0, w0} cannot be center-cropped to {target_shape}.",
            )
        d_start = (d0 - d_t) // 2
        h_start = (h0 - h_t) // 2
        w_start = (w0 - w_t) // 2
        return grid[
            :,
            d_start : d_start + d_t,
            h_start : h_start + h_t,
            w_start : w_start + w_t,
            :,
        ]

    if pts_world.ndim == 3:
        batch_size, num_pts, _ = pts_world.shape
        pts_shape = _infer_pts_shape(int(num_pts), grid_shape)
        pts_grid = pts_world.view(
            batch_size,
            pts_shape[0],
            pts_shape[1],
            pts_shape[2],
            3,
        )
        pts_grid = _center_crop(pts_grid, grid_shape)
    elif pts_world.ndim == 5:
        pts_grid = _center_crop(pts_world, grid_shape)
    else:
        raise ValueError(
            f"Expected pts_world with ndim 3 or 5, got {pts_world.ndim}.",
        )

    pts_flat = pts_grid.reshape(pts_grid.shape[0], -1, 3)

    t_rig_world = pose_world_rig_ref.inverse()
    pts_rig = t_rig_world * pts_flat

    extent = voxel_extent.to(device=pts_rig.device, dtype=pts_rig.dtype)
    if extent.ndim == 1:
        extent = extent.view(1, 6).expand(pts_rig.shape[0], 6)
    mins = extent[:, [0, 2, 4]]
    maxs = extent[:, [1, 3, 5]]
    center_vox = 0.5 * (mins + maxs)
    span = (maxs - mins).clamp_min(1e-6)
    scale = 0.5 * span

    center_vox = center_vox[:, None, :]
    center_world = (t_world_voxel * center_vox).squeeze(1)
    center_rig = (t_rig_world * center_world[:, None, :]).squeeze(1)
    pts_norm = (pts_rig - center_rig[:, None, :]) / scale[:, None, :]

    pts_norm = pts_norm.view(
        pts_grid.shape[0],
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        3,
    )
    return pts_norm.permute(0, 4, 1, 2, 3).contiguous()

"""Vectorised depth→point-cloud conversion for candidate renders.

Consumes a single :class:`EfmSnippetView` and matching :class:`CandidateDepths`
to produce padded per-candidate point clouds, fused clouds with the collapsed
semi-dense SLAM reconstruction, and a combined occupancy extent for cropping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from efm3d.aria import PoseTW
from pydantic import Field

from ..data_handling import EfmSnippetView
from ..utils import BaseConfig
from ..utils.mojo_subprocess import run_mojo_subprocess
from ..utils.pytorch3d_compat import PerspectiveCameras
from ..utils.typed_payloads import from_serializable, to_serializable
from .camera_batches import NativeCameraBatch, is_native_camera_batch, require_pytorch3d_camera_batch
from .candidate_depth_renderer import CandidateDepths
from .mojo_backend import is_mojo_thread_context_supported, unproject_candidate_points_mojo

Tensor = torch.Tensor


class PointCloudBackend(StrEnum):
    """Backend selector for candidate point-cloud construction."""

    PYTORCH3D = "pytorch3d"
    MOJO = "mojo"


class MojoPointCloudBuilderConfig(BaseConfig):
    """Nested config for the Mojo point-cloud builder path."""

    workers: int | None = None
    """Optional worker override for the Mojo point-cloud kernel."""


class CandidatePointCloudBuilderConfig(BaseConfig):
    """Config-as-factory wrapper for :class:`CandidatePointCloudBuilder`."""

    @property
    def target(self) -> type["CandidatePointCloudBuilder"]:
        return CandidatePointCloudBuilder

    backend: PointCloudBackend = PointCloudBackend.PYTORCH3D
    """Backend used for depth backprojection and compaction."""

    backprojection_stride: int = 1
    """Pixel stride used when backprojecting depth maps."""

    mojo: MojoPointCloudBuilderConfig = Field(default_factory=MojoPointCloudBuilderConfig)
    """Nested config for the Mojo path."""


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

    def to_serializable(self) -> dict[str, object]:
        """Serialize this point-cloud batch into a cache-friendly CPU payload."""

        return to_serializable(self)

    @classmethod
    def from_serializable(
        cls,
        payload: dict[str, object],
        *,
        device: torch.device,
    ) -> "CandidatePointClouds":
        """Reconstruct one point-cloud batch from a serialized payload.

        Args:
            payload: Serialized payload produced by :meth:`to_serializable`.
            device: Destination device for tensors.

        Returns:
            Reconstructed candidate-pointcloud batch.
        """

        return from_serializable(cls, payload, device=device)


class CandidatePointCloudBuilder:
    """Backend-driven builder for candidate point clouds."""

    config: CandidatePointCloudBuilderConfig

    def __init__(self, config: CandidatePointCloudBuilderConfig) -> None:
        self.config = config

    def build(
        self,
        sample: EfmSnippetView,
        batch: CandidateDepths,
    ) -> CandidatePointClouds:
        """Convert stacked depth maps into batched point clouds and fuse with SLAM."""

        depths = batch.depths
        if depths.ndim != 3:
            raise ValueError(f"Expected depths of shape (B,H,W), got {tuple(depths.shape)}")

        if self.config.backend == PointCloudBackend.PYTORCH3D:
            camera_batch = batch.resolved_camera_batch()
            if camera_batch is None:
                raise ValueError("PyTorch3D point-cloud backend requires a camera batch.")
            cameras = require_pytorch3d_camera_batch(camera_batch)
            padded, lengths = _backproject_depths_p3d_batch(
                depths=depths,
                mask_valid=batch.depths_valid_mask,
                cameras=cameras,
                stride=int(self.config.backprojection_stride),
            )
        else:
            camera_batch = batch.resolved_camera_batch()
            if camera_batch is None:
                raise ValueError("Mojo point-cloud backend requires a camera batch.")
            if not is_mojo_thread_context_supported():
                if not is_native_camera_batch(camera_batch):
                    raise TypeError("Thread-compatible Mojo point-cloud fallback requires a native camera batch.")
                payload = {
                    "depths": depths.detach().to(device="cpu"),
                    "mask_valid": batch.depths_valid_mask.detach().to(device="cpu"),
                    "poses": batch.poses.tensor().detach().to(device="cpu"),
                    "camera": camera_batch.camera_tw.tensor().detach().to(device="cpu"),
                    "stride": int(self.config.backprojection_stride),
                }
                result = run_mojo_subprocess("unproject_points", payload)
                padded = result["points"].to(device=depths.device, dtype=depths.dtype)
                lengths = result["lengths"].to(device=depths.device)
            else:
                padded, lengths = _backproject_depths_mojo_batch(
                    depths=depths,
                    mask_valid=batch.depths_valid_mask,
                    poses=batch.poses,
                    camera_batch=camera_batch,
                    stride=int(self.config.backprojection_stride),
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


def build_candidate_pointclouds(
    sample: EfmSnippetView,
    batch: CandidateDepths,
    *,
    stride: int = 1,
) -> CandidatePointClouds:
    """Compatibility helper that defaults to the current PyTorch3D backend."""

    cfg = CandidatePointCloudBuilderConfig(backprojection_stride=int(stride))
    return cfg.setup_target().build(sample, batch)


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


def _backproject_depths_mojo_batch(
    depths: Tensor,
    mask_valid: Tensor,
    poses: PoseTW,
    camera_batch: PerspectiveCameras | NativeCameraBatch,
    *,
    stride: int = 1,
) -> tuple[Tensor, Tensor]:
    """Backproject stacked depth maps via the Mojo point-cloud kernel."""

    bsz = int(depths.shape[0])
    points_all: list[Tensor] = []
    lengths: list[int] = []
    for idx in range(bsz):
        pose_i = poses if poses.tensor().ndim == 1 else poses[idx]
        if is_native_camera_batch(camera_batch):
            camera_i = (
                camera_batch.camera_tw if camera_batch.camera_tw.tensor().ndim == 1 else camera_batch.camera_tw[idx]
            )
        else:
            raise TypeError("Mojo point-cloud backend requires a native camera batch.")
        points_i, _ = unproject_candidate_points_mojo(
            depths[idx],
            mask_valid[idx],
            pose_world_cam=pose_i,
            camera=camera_i,
            stride=stride,
            device=depths.device,
        )
        points_all.append(points_i)
        lengths.append(int(points_i.shape[0]))

    lengths_t = torch.tensor(lengths, device=depths.device, dtype=torch.long)
    max_len = int(lengths_t.max().item()) if lengths_t.numel() > 0 else 0
    if max_len == 0:
        return torch.empty((bsz, 0, 3), device=depths.device, dtype=depths.dtype), lengths_t

    padded = torch.full((bsz, max_len, 3), torch.nan, device=depths.device, dtype=depths.dtype)
    for idx, points_i in enumerate(points_all):
        if points_i.numel() == 0:
            continue
        padded[idx, : points_i.shape[0]] = points_i.to(device=depths.device, dtype=depths.dtype)
    return padded, lengths_t


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


__all__ = [
    "CandidatePointCloudBuilder",
    "CandidatePointCloudBuilderConfig",
    "CandidatePointClouds",
    "MojoPointCloudBuilderConfig",
    "PointCloudBackend",
    "build_candidate_pointclouds",
]

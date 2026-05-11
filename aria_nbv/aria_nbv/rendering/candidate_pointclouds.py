"""Vectorised depth-to-point-cloud conversion for candidate renders.

Consumes a single `EfmSnippetView` and matching `CandidateDepths`
to produce padded per-candidate point clouds, fused clouds with the collapsed
semi-dense SLAM reconstruction, and a combined occupancy extent for cropping.

The output is world-frame oracle evidence. Scene-level RRI uses the combined
snippet/candidate extent; target-level RRI may further crop both candidate
points and mesh geometry with the matched GT target OBB. Empty target crops or
unusable candidate depth should surface as invalidity, not as a low score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_handling import EfmSnippetView
from ..utils.typed_payloads import from_serializable, to_serializable
from .candidate_depth_renderer import CandidateDepths
from .unproject import backproject_depths_p3d_batch

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
            payload: Serialized payload produced by `to_serializable`.
            device: Destination device for tensors.

        Returns:
            Reconstructed candidate-pointcloud batch.
        """

        return from_serializable(cls, payload, device=device)


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

    padded, lengths = backproject_depths_p3d_batch(
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

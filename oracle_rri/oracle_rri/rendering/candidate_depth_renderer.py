"""Utilities to render candidate poses directly from dataset snippets."""

from __future__ import annotations

from typing import Literal, TypedDict

import torch
from efm3d.aria import CameraTW, PoseTW
from pydantic import Field

from ..data.efm_views import EfmCameraView, EfmSnippetView
from ..pose_generation.types import CandidateSamplingResult
from ..utils import BaseConfig, Console
from .pytorch3d_depth_renderer import Pytorch3DDepthRendererConfig

CameraStream = Literal["rgb", "rgb_depth", "slam_left", "slam_right"]
FrameSelection = Literal["first", "last"]


class CandidateDepthBatch(TypedDict):
    """Typed result for candidate depth rendering."""

    depths: torch.Tensor
    """Tensor['N', 'H', 'W'] with per-candidate depth maps in metres."""

    poses: "PoseTW"
    """PoseTW with the subset of candidates that were rendered."""

    mask_valid: torch.Tensor
    """Boolean mask from :class:`CandidateSamplingResult` (unchanged, 1-D)."""

    candidate_indices: torch.Tensor
    """Indices (long) into the original candidate array corresponding to ``depths``."""

    camera: "CameraTW"
    """Camera calibration used for rendering (single frame)."""


class CandidateDepthRendererConfig(BaseConfig["CandidateDepthRenderer"]):
    """Config-as-factory wrapper for :class:`CandidateDepthRenderer`."""

    target: type["CandidateDepthRenderer"] = Field(
        default_factory=lambda: CandidateDepthRenderer,
        exclude=True,
    )
    """Factory target for :meth:`BaseConfig.setup_target`."""

    renderer: Pytorch3DDepthRendererConfig = Field(default_factory=Pytorch3DDepthRendererConfig)
    """Nested config describing the underlying PyTorch3D renderer."""

    camera_stream: CameraStream = "rgb"
    """Which camera stream to pull calibration from."""

    frame_selection: FrameSelection = "last"
    """Which frame from the snippet to use for intrinsics/extrinsics (first/last)."""

    max_candidates: int | None = None
    """Optional cap on number of valid candidates rendered per call."""

    verbose: bool = False
    """Enable structured logging."""

    is_debug: bool = False
    """Enable detailed debug logging."""


class CandidateDepthRenderer:
    """High-level helper that renders depth for candidate poses."""

    config: CandidateDepthRendererConfig

    def __init__(self, config: CandidateDepthRendererConfig) -> None:
        self.config = config
        debug_flag = self.config.is_debug or getattr(self.config.renderer, "is_debug", False)
        self.console = (
            Console.with_prefix(self.__class__.__name__).set_verbose(self.config.verbose).set_debug(debug_flag)
        )
        self.renderer = self.config.renderer.setup_target()

    def render(
        self,
        sample: EfmSnippetView,
        candidates: CandidateSamplingResult,
    ) -> CandidateDepthBatch:
        """Render depth maps for valid candidate poses within a snippet."""

        if not sample.has_mesh or sample.mesh is None:
            msg = "CandidateDepthRenderer requires snippets with attached meshes."
            self.console.error(msg)
            raise ValueError(msg)
        self.console.log(
            f"Rendering depths: candidates={candidates['poses'].tensor().shape[0]} stream={self.config.camera_stream}"
        )
        self.console.log(f"Mesh stats verts={sample.mesh.vertices.shape[0]:,} faces={sample.mesh.faces.shape[0]:,}")
        pose_batch, mask_valid, candidate_indices = self._filter_candidates(candidates)
        self.console.log_summary("candidate_indices", candidate_indices)
        camera_view = self._camera_view(sample)
        frame_idx = 0 if self.config.frame_selection == "first" else -1
        frame_calib = self._frame_calib(camera_view.calib, frame_idx)
        self.console.log_summary("camera_calib_frame", frame_calib.tensor())
        self.console.dbg_summary("pose_batch_tensor", pose_batch.tensor())
        occ_extent = self._occupancy_extent(sample, device=pose_batch.tensor().device)
        depths = self.renderer.render_batch(
            poses=pose_batch,
            mesh=sample.mesh,
            camera=camera_view.calib,
            frame_index=frame_idx,
            occupancy_extent=occ_extent,
        )
        hit_ratio = float((depths < self.renderer.config.zfar).float().mean().item())
        self.console.log_summary("depth_batch_stats", {"hit_ratio": hit_ratio, "zfar": self.renderer.config.zfar})
        if hit_ratio == 0.0:
            self.console.warn(
                "All candidate depth pixels are at zfar; check camera poses, mesh normals, or backface culling."
            )
        return {
            "depths": depths,
            "poses": pose_batch,
            "mask_valid": mask_valid,
            "candidate_indices": candidate_indices,
            "camera": frame_calib,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _camera_view(self, sample: EfmSnippetView) -> EfmCameraView:
        stream = self.config.camera_stream
        if stream in ("rgb", "rgb_depth"):
            return sample.camera_rgb
        if stream == "slam_left":
            return sample.camera_slam_left
        if stream == "slam_right":
            return sample.camera_slam_right
        raise ValueError(f"Unsupported camera stream '{stream}'.")

    def _frame_calib(self, calib: CameraTW, frame_idx: int) -> CameraTW:
        tensor = calib.tensor()
        if tensor.ndim == 1:
            return calib
        frame = tensor[frame_idx].unsqueeze(0)
        return CameraTW(frame)

    def _filter_candidates(
        self,
        candidates: CandidateSamplingResult,
    ) -> tuple["PoseTW", torch.Tensor, torch.Tensor]:
        poses = candidates["poses"]
        mask_valid = candidates.get("mask_valid")
        if mask_valid is None:
            mask_valid = torch.ones(len(poses), dtype=torch.bool, device=poses.tensor().device)
        valid_idx = torch.nonzero(mask_valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            self.console.warn("No valid candidates available for rendering.")
            raise ValueError("No valid candidates to render.")
        if self.config.max_candidates is not None:
            valid_idx = valid_idx[: self.config.max_candidates]

        pose_tensor = poses.tensor()[valid_idx]
        pose_batch = PoseTW(pose_tensor)
        return pose_batch, mask_valid, valid_idx

    def _occupancy_extent(self, sample: EfmSnippetView, *, device: torch.device) -> torch.Tensor | None:
        """Return ``[xmin, xmax, ymin, ymax, zmin, zmax]`` bounds from semidense metadata."""

        volume_min = getattr(sample.semidense, "volume_min", None)
        volume_max = getattr(sample.semidense, "volume_max", None)
        if volume_min is None or volume_max is None:
            return None
        vmin = volume_min.to(device=device, dtype=torch.float32)
        vmax = volume_max.to(device=device, dtype=torch.float32)
        return torch.stack([vmin[0], vmax[0], vmin[1], vmax[1], vmin[2], vmax[2]], dim=0)


__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
]

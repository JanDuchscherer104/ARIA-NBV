"""Utilities to render candidate poses directly from dataset snippets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

import torch
from efm3d.aria import PoseTW
from pydantic import Field

from ..data.efm_views import EfmCameraView, EfmSnippetView
from ..pose_generation.types import CandidateSamplingResult
from ..utils import BaseConfig, Console
from .pytorch3d_depth_renderer import Pytorch3DDepthRendererConfig

if TYPE_CHECKING:
    from efm3d.aria import CameraTW, PoseTW

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


class CandidateDepthRenderer:
    """High-level helper that renders depth for candidate poses."""

    config: CandidateDepthRendererConfig

    def __init__(self, config: CandidateDepthRendererConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(self.config.verbose)
        self.renderer = self.config.renderer.setup_target()

    def render(
        self,
        sample: EfmSnippetView,
        candidates: CandidateSamplingResult,
    ) -> CandidateDepthBatch:
        """Render depth maps for valid candidate poses within a snippet."""

        if not sample.has_mesh or sample.mesh is None:
            raise ValueError("CandidateDepthRenderer requires snippets with attached meshes.")
        pose_batch, mask_valid, candidate_indices = self._filter_candidates(candidates)
        camera_view = self._camera_view(sample)
        frame_idx = 0 if self.config.frame_selection == "first" else -1
        depths = self.renderer.render_batch(
            poses=pose_batch,
            mesh=sample.mesh,
            camera=camera_view.calib,
            frame_index=frame_idx,
        )
        camera_calib = camera_view.calib[frame_idx]
        return {
            "depths": depths,
            "poses": pose_batch,
            "mask_valid": mask_valid,
            "candidate_indices": candidate_indices,
            "camera": camera_calib,
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
            raise ValueError("No valid candidates to render.")
        if self.config.max_candidates is not None:
            valid_idx = valid_idx[: self.config.max_candidates]

        pose_tensor = poses.tensor()[valid_idx]
        pose_batch = PoseTW(pose_tensor)
        return pose_batch, mask_valid, valid_idx


__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
]

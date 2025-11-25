"""Utilities to render candidate poses directly from dataset snippets."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import torch
from efm3d.aria import CameraTW, PoseTW
from pydantic import AliasChoices, Field, field_validator, model_validator

from ..data.efm_views import EfmCameraView, EfmSnippetView
from ..pose_generation.types import CandidateSamplingResult
from ..utils import BaseConfig, Console, Verbosity, pick_fast_depth_renderer
from .efm3d_depth_renderer import Efm3dDepthRendererConfig
from .pytorch3d_depth_renderer import Pytorch3DDepthRendererConfig

CameraStream = Literal["rgb", "slaml", "slamr"]
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


RendererConfig = Pytorch3DDepthRendererConfig | Efm3dDepthRendererConfig


class CandidateDepthRendererConfig(BaseConfig["CandidateDepthRenderer"]):
    """Config-as-factory wrapper for :class:`CandidateDepthRenderer`."""

    target: type["CandidateDepthRenderer"] = Field(
        default_factory=lambda: CandidateDepthRenderer,
        exclude=True,
    )
    """Factory target for :meth:`BaseConfig.setup_target`."""

    renderer: RendererConfig = Field(default_factory=Pytorch3DDepthRendererConfig)
    """Nested config describing the underlying renderer (PyTorch3D or CPU)."""

    camera_stream: CameraStream = "rgb"
    """Which camera stream to pull calibration from."""

    frame_selection: FrameSelection = "last"
    """Which frame from the snippet to use for intrinsics/extrinsics (first/last)."""

    max_candidates: int | None = None
    """Optional cap on number of valid candidates rendered per call."""

    resolution_scale: float | None = None
    """Optional uniform scale (0<scale<=1) applied to H,W for rendering. Ignored if ``low_res`` is True."""

    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging (0=quiet, 1=normal, 2=verbose).",
    )
    """Verbosity level for logging."""

    is_debug: bool = False
    """Enable detailed debug logging."""

    @model_validator(mode="after")
    def _apply_performance_mode(self) -> "CandidateDepthRendererConfig":
        upgraded = pick_fast_depth_renderer(self.renderer)
        object.__setattr__(self, "renderer", upgraded)
        return self

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Any) -> Verbosity:
        return Verbosity.from_any(value)


class CandidateDepthRenderer:
    """High-level helper that renders depth for candidate poses."""

    config: CandidateDepthRendererConfig

    def __init__(self, config: CandidateDepthRendererConfig) -> None:
        self.config = config
        debug_flag = self.config.is_debug or getattr(self.config.renderer, "is_debug", False)
        self.console = (
            Console.with_prefix(self.__class__.__name__).set_verbosity(self.config.verbosity).set_debug(debug_flag)
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
            f"Rendering depths: candidates={candidates['poses'].tensor().shape[0]} stream={self.config.camera_stream} on '{self.renderer.__class__.__name__}'"
        )
        self.console.log(f"Mesh stats verts={sample.mesh.vertices.shape[0]:,} faces={sample.mesh.faces.shape[0]:,}")
        pose_batch, mask_valid, candidate_indices = self._filter_candidates(candidates)
        self.console.log(f"Attempting renders for {pose_batch.tensor().shape[0]} candidates (GUI slice).")
        self.console.log_summary("candidate_indices", candidate_indices)

        camera_view = sample.get_camera(self.config.camera_stream)
        if self.config.resolution_scale is not None:
            self.console.log(f"Downscaling camera view by scale {self.config.resolution_scale}")
            camera_view = self._downscale_camera_view(camera_view, scale=self.config.resolution_scale)
        frame_idx = 0 if self.config.frame_selection == "first" else -1
        frame_calib = self._frame_calib(camera_view.calib, frame_idx)
        self.console.log_summary("camera_calib_frame", frame_calib.tensor())
        self.console.dbg_summary("pose_batch_tensor", pose_batch.tensor())
        if hasattr(self.renderer, "render_batch"):
            depths = self.renderer.render_batch(
                poses=pose_batch,
                mesh=sample.mesh,
                camera=frame_calib,
                frame_index=frame_idx,
            )
        else:
            depth_list = []
            for pose in pose_batch:  # type: ignore[attr-defined]
                depth_i = self.renderer.render_depth(
                    pose_world_cam=pose,
                    mesh=sample.mesh,
                    camera=camera_view.calib,
                    frame_index=frame_idx,
                )
                depth_list.append(depth_i)
            depths = torch.stack(depth_list, dim=0)
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

    def _frame_calib(self, calib: CameraTW, frame_idx: int) -> CameraTW:
        tensor = calib.tensor()
        if tensor.ndim == 1:
            return calib
        frame = tensor[frame_idx].unsqueeze(0)
        cam = CameraTW(frame)
        # TODO: Potential error here!

        # Candidate poses already encode world←camera; avoid double-applying dataset extrinsics
        cam.set_T_camera_rig(
            PoseTW.from_Rt(
                torch.eye(3, device=cam.device, dtype=cam.dtype), torch.zeros(3, device=cam.device, dtype=cam.dtype)
            )
        )

        return cam

    def _downscale_camera_view(
        self,
        view: EfmCameraView,
        *,
        scale: float | None = None,
    ) -> EfmCameraView:
        """Downscale images and intrinsics for faster rendering."""

        _, _, h, w = view.images.shape
        if scale is not None and scale != 1.0:
            new_w = max(64, int(w * scale))
            new_h = max(64, int(h * scale))
        else:
            return view

        with torch.no_grad():
            images_small = torch.nn.functional.interpolate(
                view.images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        return EfmCameraView(
            images=images_small,
            calib=view.calib.scale_to_size((new_w, new_h)),  # type: ignore
            time_ns=view.time_ns,
            frame_ids=view.frame_ids,
        )

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


__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
]

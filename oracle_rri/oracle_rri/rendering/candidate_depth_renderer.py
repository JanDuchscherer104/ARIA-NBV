"""Utilities to render candidate poses directly from dataset snippets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from efm3d.aria import CameraTW, PoseTW
from pydantic import AliasChoices, Field, field_validator, model_validator

from ..data.efm_views import EfmSnippetView
from ..data.mesh_cache import mesh_from_snippet
from ..pose_generation.types import CandidateSamplingResult
from ..utils import BaseConfig, Console, Verbosity, pick_fast_depth_renderer
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig


@dataclass(slots=True)
class CandidateDepthBatch:
    """Typed result for candidate depth rendering."""

    depths: torch.Tensor
    """Tensor['N', 'H', 'W'] with per-candidate depth maps in metres."""

    poses: PoseTW
    """PoseTW (world←camera) for the rendered subset."""

    mask_valid: torch.Tensor
    """Boolean mask aligned with ``candidate_indices`` (subset of original mask_valid)."""

    candidate_indices: torch.Tensor
    """Indices (long) into the original candidate array corresponding to ``depths``."""

    camera: CameraTW
    """Camera calibration (and ref←cam extrinsics) for the rendered subset, same order as ``depths``."""


class CandidateDepthRendererConfig(BaseConfig["CandidateDepthRenderer"]):
    """Config-as-factory wrapper for :class:`CandidateDepthRenderer`."""

    target: type["CandidateDepthRenderer"] = Field(
        default_factory=lambda: CandidateDepthRenderer,
        exclude=True,
    )
    """Factory target for :meth:`BaseConfig.setup_target`."""

    renderer: Pytorch3DDepthRendererConfig = Field(default_factory=Pytorch3DDepthRendererConfig)
    """Nested config describing the underlying renderer (PyTorch3D or CPU)."""

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
            f"Rendering depths: candidates={candidates.views.tensor().shape[0]} on '{self.renderer.__class__.__name__}'"
        )
        self.console.log(f"Mesh stats verts={sample.mesh.vertices.shape[0]:,} faces={sample.mesh.faces.shape[0]:,}")
        pose_batch, camera_batch, candidate_indices = self._select_candidates(candidates)
        self.console.log(f"Attempting renders for {pose_batch.tensor().shape[0]} candidates (GUI slice).")
        self.console.dbg_summary("candidate_indices", candidate_indices)

        camera_calib = camera_batch
        if self.config.resolution_scale is not None:
            scale = float(self.config.resolution_scale)
            self.console.log(f"Scaling candidate camera intrinsics by {scale}")
            base_size = camera_calib.size[0]
            new_size = (int(base_size[0].item() * scale), int(base_size[1].item() * scale))
            camera_calib = camera_calib.scale_to_size(new_size)

        self.console.dbg_summary("camera_calib_batch", camera_calib)
        self.console.dbg_summary("pose_batch_tensor", pose_batch)
        is_pytorch3d = isinstance(self.renderer, Pytorch3DDepthRenderer)
        mesh_input: Any
        if is_pytorch3d and sample.mesh_verts is not None and sample.mesh_faces is not None:
            mesh_input = (sample.mesh_verts, sample.mesh_faces)
        elif is_pytorch3d and sample.mesh is not None:
            mesh_input = (
                torch.as_tensor(sample.mesh.vertices, device=pose_batch.device, dtype=torch.float32),
                torch.as_tensor(sample.mesh.faces, device=pose_batch.device, dtype=torch.int64),
            )
        elif is_pytorch3d:
            art = mesh_from_snippet(sample, device=pose_batch.device, console=self.console)
            mesh_input = (art.processed.verts, art.processed.faces)
        else:
            mesh_input = sample.mesh

        if not isinstance(self.renderer, Pytorch3DDepthRenderer):
            raise TypeError(f"Unsupported renderer type: {self.renderer.__class__.__name__}")

        depths = self.renderer.render(
            poses=pose_batch,
            mesh=mesh_input,
            camera=camera_calib,
            frame_index=None,
            mesh_cache_key=sample.mesh_cache_key,
        )

        hit_ratio = float((depths < self.renderer.config.zfar).float().mean().item())
        self.console.dbg_summary("depth_batch_stats", {"hit_ratio": hit_ratio, "zfar": self.renderer.config.zfar})
        if hit_ratio == 0.0:
            self.console.warn(
                "All candidate depth pixels are at zfar; check camera poses, mesh normals, or backface culling."
            )
        mask_valid_subset = candidates.mask_valid.to(device=pose_batch.device)[candidate_indices]
        return CandidateDepthBatch(
            depths=depths,
            poses=pose_batch,
            mask_valid=mask_valid_subset,
            candidate_indices=candidate_indices,
            camera=camera_calib,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_candidates(
        self,
        candidates: CandidateSamplingResult,
    ) -> tuple["PoseTW", CameraTW, torch.Tensor]:
        device = self.renderer.device
        cam_views = candidates.views.to(device)
        num_candidates = cam_views.tensor().shape[0]
        if num_candidates == 0:
            raise ValueError("No candidates provided for rendering.")

        if self.config.max_candidates is not None:
            k = min(num_candidates, self.config.max_candidates)
            candidate_idx = torch.arange(k, device=device, dtype=torch.long)
        else:
            candidate_idx = torch.arange(num_candidates, device=device, dtype=torch.long)

        selected_views = cam_views[candidate_idx]
        # CandidateSamplingResult convention: T_camera_rig is reference<-camera.
        t_ref_cam = selected_views.T_camera_rig.to(device=device)  # reference ← camera
        t_world_ref = candidates.reference_pose.to(device=device)  # world ← reference
        poses_world_cam = t_world_ref @ t_ref_cam  # world ← camera

        return poses_world_cam, selected_views, candidate_idx


__all__ = [
    "CandidateDepthBatch",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
]

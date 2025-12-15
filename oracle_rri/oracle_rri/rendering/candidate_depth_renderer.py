"""Utilities to render candidate poses directly from dataset snippets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import torch
from efm3d.aria import CameraTW, PoseTW
from pydantic import Field, field_validator
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from ..data.efm_views import EfmSnippetView
from ..pose_generation.types import CandidateSamplingResult
from ..utils import BaseConfig, Console, Verbosity
from .pytorch3d_depth_renderer import Pytorch3DDepthRenderer, Pytorch3DDepthRendererConfig


@dataclass(slots=True)
class CandidateDepths:
    """Typed result for candidate depth rendering."""

    depths: torch.Tensor
    """Tensor['N', 'H', 'W'] with per-candidate depth maps in metres."""

    depths_valid_mask: torch.Tensor
    """Tensor['N', 'H', 'W'] boolean mask indicating valid depth pixels."""

    poses: PoseTW
    """PoseTW (cam2world) for the rendered subset."""

    reference_pose: PoseTW
    """PoseTW (ref2world) for the reference frame corresponding to candidates."""

    candidate_indices: torch.Tensor
    """Indices (long) into the **full** candidate array (pre-render filtering)."""

    camera: CameraTW
    """Camera calibration (and ref2camera extrinsics) for the rendered subset, same order as ``depths``."""

    p3d_cameras: PerspectiveCameras
    """PyTorch3D PerspectiveCameras used for rendering. Same externals as ``camera``."""


class CandidateDepthRendererConfig(BaseConfig["CandidateDepthRenderer"]):
    target: type["CandidateDepthRenderer"] = Field(
        default_factory=lambda: CandidateDepthRenderer,
        exclude=True,
    )

    renderer: Pytorch3DDepthRendererConfig = Field(default_factory=Pytorch3DDepthRendererConfig)
    """Nested config describing the underlying renderer (PyTorch3D or CPU)."""

    max_candidates_final: int = 16
    """Number of valid candidates after oversampling and filtering."""

    oversample_factor: float = 2.0

    resolution_scale: float | None = None
    """Optional uniform scale (0<scale<=1) applied to H,W for rendering. Ignored if ``low_res`` is True."""

    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
    )
    """Verbosity level for logging."""

    is_debug: bool = False
    """Enable detailed debug logging."""

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
    ) -> CandidateDepths:
        """Render depth maps for valid candidate poses within a snippet."""

        if not sample.has_mesh or sample.mesh is None:
            msg = "CandidateDepthRenderer requires snippets with attached meshes."
            self.console.error(msg)
            raise ValueError(msg)
        self.console.log(
            f"Rendering depths: candidates={candidates.views.tensor().shape[0]} on '{self.renderer.__class__.__name__}'"
        )
        self.console.log(f"Mesh stats verts={sample.mesh.vertices.shape[0]:,} faces={sample.mesh.faces.shape[0]:,}")
        pose_batch, camera_calib, candidate_indices = self._select_candidate_views(candidates)
        self.console.log(f"Attempting renders for {pose_batch.tensor().shape[0]} candidates (GUI slice).")
        self.console.dbg_summary("candidate_indices", candidate_indices)

        if self.config.resolution_scale is not None:
            scale = float(self.config.resolution_scale)
            self.console.log(f"Scaling candidate camera intrinsics by {scale}")
            base_size = camera_calib.size[0]
            new_size = (int(base_size[0].item() * scale), int(base_size[1].item() * scale))
            camera_calib = camera_calib.scale_to_size(new_size)

        self.console.dbg_summary("camera_calib_batch", camera_calib)
        self.console.dbg_summary("pose_batch_tensor", pose_batch)
        if not isinstance(self.renderer, Pytorch3DDepthRenderer):
            raise TypeError(f"Unsupported renderer type: {self.renderer.__class__.__name__}")

        depths, pix_to_face, cameras = self.renderer.render(
            poses=pose_batch,
            mesh=(sample.mesh_verts, sample.mesh_faces),
            camera=camera_calib,
            frame_index=None,
        )

        self.console.dbg(
            f"Rendered {depths.shape[0]} candidates | depth shape {tuple(depths.shape)} | "
            f"pix_to_face min {int(pix_to_face.min().item())} max {int(pix_to_face.max().item())}"
        )

        depths_valid_mask = (
            (pix_to_face >= 0)
            & (depths > self.renderer.config.znear * 1.01)
            & (depths < self.renderer.config.zfar * 0.99)
        )

        # (
        #     depths,
        #     depths_valid_mask,
        #     pose_batch,
        #     camera_calib,
        #     cameras,
        #     candidate_indices,
        # ) = self._filter_valid_candidates(
        #     depths=depths,
        #     depths_valid_mask=depths_valid_mask,
        #     pix_to_face=pix_to_face,
        #     pose_batch=pose_batch,
        #     camera_calib=camera_calib,
        #     cameras=cameras,
        #     candidate_indices=candidate_indices,
        # )

        return CandidateDepths(
            depths=depths,
            depths_valid_mask=depths_valid_mask,
            poses=pose_batch,
            reference_pose=candidates.reference_pose.to(device=pose_batch.device),
            candidate_indices=candidate_indices,
            camera=camera_calib,
            p3d_cameras=cameras,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_candidate_views(
        self,
        candidates: CandidateSamplingResult,
    ) -> tuple[PoseTW, CameraTW, torch.Tensor]:
        """
        Select a subset of candidate poses for rendering, based on config.

        Returns:
            - PoseTW: Selected candidate poses (world2cam).
            - CameraTW: Selected candidate cameras (extrinsics are ref2cam).
            - Tensor['K',]: Indices into original candidate array.
        """
        device = self.renderer.device
        cam_views = candidates.views.to(device)
        num_candidates = cam_views.tensor().shape[0]
        # Map rendered subset back to the original (pre-pruning) candidate order so that
        # dashboard indices stay aligned with sampling diagnostics.
        if candidates.mask_valid is None or candidates.shell_poses is None:
            valid_global_idx = torch.arange(num_candidates, device=device, dtype=torch.long)
        else:
            valid_mask = candidates.mask_valid.to(device)
            valid_global_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            if valid_global_idx.numel() != num_candidates:
                # Fallback to sequential if something went wrong; keeps ordering stable.
                valid_global_idx = torch.arange(num_candidates, device=device, dtype=torch.long)
        if num_candidates == 0:
            raise ValueError("No candidates provided for rendering.")

        num_render = min(
            num_candidates,
            max(1, ceil(self.config.oversample_factor * self.config.max_candidates_final)),
        )

        candidate_idx = torch.arange(num_render, device=device, dtype=torch.long)

        selected_views = cam_views[candidate_idx]
        poses_world_cam = candidates.poses_world_cam(device=device)[candidate_idx]

        return poses_world_cam, selected_views, valid_global_idx[candidate_idx]

    def _filter_valid_candidates(
        self,
        *,
        depths: torch.Tensor,
        depths_valid_mask: torch.Tensor,
        pix_to_face: torch.Tensor,
        pose_batch: PoseTW,
        camera_calib: CameraTW,
        cameras: PerspectiveCameras,
        candidate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PoseTW, CameraTW, PerspectiveCameras, torch.Tensor]:
        """Keep only renders whose pix-to-face map has no misses and cap output size.

        A candidate render is considered valid only if *all* pixels hit a mesh
        triangle, i.e. its ``pix_to_face`` slice contains no negative entries.
        Oversampling in :meth:`_select_candidate_views` ensures we can still
        return up to ``max_candidates_final`` valid renders after filtering.
        """

        candidate_valid = self._candidate_hit_mask(pix_to_face)
        num_valid = int(candidate_valid.sum().item())
        num_total = candidate_valid.numel()

        self.console.dbg(f"Candidate validity (no misses): {num_valid}/{num_total}")

        if num_valid == 0:
            msg = "No valid candidate renders: all pix_to_face maps contain misses."
            self.console.error(msg)
            raise ValueError(msg)

        keep_idx = torch.nonzero(candidate_valid, as_tuple=False).squeeze(-1)
        if keep_idx.numel() > self.config.max_candidates_final:
            discarded_valid = keep_idx.numel() - self.config.max_candidates_final
            keep_idx = keep_idx[: self.config.max_candidates_final]
            self.console.dbg(
                f"Discarding {discarded_valid} valid renders after capping to max_candidates_final="
                f"{self.config.max_candidates_final}"
            )

        if keep_idx.numel() < self.config.max_candidates_final:
            self.console.warn(
                f"Only {keep_idx.numel()} valid renders out of {num_total}; requested max {self.config.max_candidates_final}."
            )
        elif num_valid > keep_idx.numel():
            self.console.log(f"Filtered {num_valid - keep_idx.numel()} renders after oversampling ({num_total} total).")

        return (
            depths[keep_idx],
            depths_valid_mask[keep_idx],
            pose_batch[keep_idx],
            camera_calib[keep_idx],
            cameras[keep_idx],
            candidate_indices[keep_idx],
        )

    @staticmethod
    def _candidate_hit_mask(pix_to_face: torch.Tensor) -> torch.Tensor:
        """True for candidates whose pix_to_face has no negative entries."""

        flat = pix_to_face.view(pix_to_face.shape[0], -1)
        return flat.min(dim=1).values >= 0


__all__ = [
    "CandidateDepths",
    "CandidateDepthRenderer",
    "CandidateDepthRendererConfig",
]

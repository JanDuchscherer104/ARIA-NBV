"""Modular VIN pipeline (refactored) for improved separation of concerns.

This module provides a drop-in alternative to ``vin.model.VinModel`` that
decomposes the forward pass into focused components:

- Pose encoding (candidate + voxel shells)
- Scene-field construction (EVL evidence -> compact voxel field)
- Frustum sampling (candidate-local voxel queries)
- Feature assembly + scoring (CORAL ordinal head)

It intentionally reuses the same math as ``vin.model`` so outputs can be
compared for equivalence during refactor validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)
from torch import Tensor, nn

from ..rri_metrics.coral import coral_expected_from_logits, coral_logits_to_prob
from ..utils import BaseConfig
from .backbone_evl import EvlBackboneConfig
from .model import (
    VinScorerHeadConfig,
    _build_frustum_points_world_p3d,
    _build_scene_field,
    _candidate_valid_from_token,
    _largest_divisor_leq,
    _sample_voxel_field,
)
from .pose_encoding import LearnableFourierFeaturesConfig
from .types import EvlBackboneOutput, VinForwardDiagnostics, VinPrediction


@dataclass(slots=True)
class PoseShellOutput:
    """Pose-encoding intermediates for a pose expressed in a reference frame."""

    center_m: Tensor
    """``Tensor["... 3", float32]`` Pose translation in the reference frame."""

    radius_m: Tensor
    """``Tensor["... 1", float32]`` Distance to the pose center."""

    center_dir: Tensor
    """``Tensor["... 3", float32]`` Unit direction to the pose center."""

    forward_dir: Tensor
    """``Tensor["... 3", float32]`` Unit forward direction in the reference frame."""

    view_alignment: Tensor
    """``Tensor["... 1", float32]`` Dot product ``<forward, -center_dir>``."""

    pose_enc: Tensor
    """``Tensor["... E", float32]`` SH-encoded shell pose descriptor."""


@dataclass(slots=True)
class SceneFieldOutput:
    """Scene-field tensors derived from EVL backbone outputs."""

    field_in: Tensor
    """``Tensor["B C_in D H W", float32]`` Raw channel field before projection."""

    field: Tensor
    """``Tensor["B C_out D H W", float32]`` Projected field used for queries."""


@dataclass(slots=True)
class FrustumOutput:
    """Frustum sampling intermediates for candidate-local queries."""

    points_world: Tensor
    """``Tensor["B N K 3", float32]`` World-space frustum points."""

    tokens: Tensor
    """``Tensor["B N K C", float32]`` Sampled voxel tokens."""

    token_valid: Tensor
    """``Tensor["B N K", bool]`` Token validity mask."""

    local_feat: Tensor
    """``Tensor["B N C", float32]`` Masked mean-pooled frustum features."""


def _ensure_candidate_batch(candidate_poses_world_cam: PoseTW) -> PoseTW:
    """Ensure candidate poses are batched as ``(B,N,12)``."""
    if candidate_poses_world_cam.ndim == 2:  # N x 12
        return PoseTW(candidate_poses_world_cam._data.unsqueeze(0))
    if candidate_poses_world_cam.ndim != 3:
        raise ValueError(
            "candidate_poses_world_cam must have shape (N,12) or (B,N,12).",
        )
    return candidate_poses_world_cam


def _ensure_pose_batch(pose: PoseTW, *, batch_size: int, name: str) -> PoseTW:
    """Broadcast a pose to ``(B,12)`` to match the candidate batch size."""
    pose_b = pose
    if pose_b.ndim == 1:
        pose_b = PoseTW(pose_b._data.unsqueeze(0))
    elif pose_b.ndim != 2:
        raise ValueError(
            f"{name} must have shape (12,) or (B,12), got ndim={pose_b.ndim}.",
        )

    if pose_b.shape[0] == 1 and batch_size > 1:
        pose_b = PoseTW(pose_b._data.expand(batch_size, 12))
    elif pose_b.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have batch size 1 or match candidate batch size.",
        )
    return pose_b


class VinPoseEncoder(nn.Module):
    """Shell-based pose encoding for candidates and voxel grid."""

    def __init__(self, config: VinPipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.pose_encoder_lff = self.config.pose_encoder_lff.setup_target()

    @property
    def out_dim(self) -> int:
        """Return the pose encoder output dimension."""
        return int(self.pose_encoder_lff.out_dim)

    def encode(self, pose_rig: PoseTW) -> PoseShellOutput:
        """Encode pose shells in a reference frame.

        Args:
            pose_rig: ``PoseTW["... 12"]`` pose in the reference frame.

        Returns:
            PoseShellOutput:
                - center_m: ``Tensor["... 3", float32]``
                - radius_m: ``Tensor["... 1", float32]``
                - center_dir: ``Tensor["... 3", float32]``
                - forward_dir: ``Tensor["... 3", float32]``
                - view_alignment: ``Tensor["... 1", float32]``
                - pose_enc: ``Tensor["... E", float32]``
        """
        center_m = pose_rig.t.to(dtype=torch.float32)
        radius_m = torch.linalg.vector_norm(center_m, dim=-1, keepdim=True)
        center_dir = center_m / (radius_m + 1e-8)

        cam_forward_axis = torch.tensor(
            [0.0, 0.0, 1.0],
            device=center_m.device,
            dtype=torch.float32,
        )
        forward_dir = torch.einsum(
            "...ij,j->...i",
            pose_rig.R.to(dtype=torch.float32),
            cam_forward_axis,
        )
        forward_dir = forward_dir / (
            torch.linalg.vector_norm(forward_dir, dim=-1, keepdim=True) + 1e-8
        )
        view_alignment = (forward_dir * (-center_dir)).sum(dim=-1, keepdim=True)

        pose_vec = torch.cat(
            [center_dir, forward_dir, radius_m, view_alignment],
            dim=-1,
        )
        pose_enc = self.pose_encoder_lff(pose_vec)
        return PoseShellOutput(
            center_m=center_m,
            radius_m=radius_m,
            center_dir=center_dir,
            forward_dir=forward_dir,
            view_alignment=view_alignment,
            pose_enc=pose_enc,
        )


class VinSceneFieldBuilder(nn.Module):
    """Build and project the compact voxel-aligned scene field."""

    def __init__(self, config: VinPipelineConfig) -> None:
        super().__init__()
        self.config = config
        field_dim = self.config.field_dim
        gn_groups = _largest_divisor_leq(field_dim, self.config.field_gn_groups)
        field_in_dim = len(self.config.scene_field_channels)

        self.field_proj = nn.Sequential(
            nn.Conv3d(field_in_dim, field_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )
        self.field_dim = field_dim

    def forward(self, backbone_out: EvlBackboneOutput) -> SceneFieldOutput:
        """Construct and project the scene field from EVL outputs.

        Args:
            backbone_out: :class:`EvlBackboneOutput` with EVL evidence volumes.

        Returns:
            SceneFieldOutput:
                - field_in: ``Tensor["B C_in D H W", float32]``
                - field: ``Tensor["B C_out D H W", float32]``
        """
        field_in = _build_scene_field(
            backbone_out,
            use_channels=self.config.scene_field_channels,
            occ_input_threshold=self.config.occ_input_threshold,
            counts_norm_mode=self.config.counts_norm_mode,
            occ_pr_is_logits=self.config.occ_pr_is_logits,
        )
        field = self.field_proj(field_in)
        return SceneFieldOutput(field_in=field_in, field=field)


class VinFrustumSampler:
    """Generate frustum points and sample a voxel field."""

    def __init__(self, config: VinPipelineConfig) -> None:
        self.config = config

    def build_points_world(
        self,
        poses_world_cam: PoseTW,
        *,
        p3d_cameras: PerspectiveCameras,
    ) -> Tensor:
        """Generate frustum points in world coordinates.

        Args:
            poses_world_cam: ``PoseTW["B N 12"]`` candidate world<-camera poses.
            p3d_cameras: PyTorch3D cameras aligned with candidates.

        Returns:
            ``Tensor["B N K 3", float32]`` world-space frustum points.
        """
        if poses_world_cam.ndim != 3:
            raise ValueError("poses_world_cam must have shape (B,N,12).")
        batch_size = int(poses_world_cam.t.shape[0])
        num_candidates = int(poses_world_cam.t.shape[1])

        cameras = p3d_cameras.to(device=poses_world_cam.t.device)
        pts_world_flat = _build_frustum_points_world_p3d(
            cameras,
            grid_size=self.config.frustum_grid_size,
            depths_m=self.config.frustum_depths_m,
        )
        num_cams = int(pts_world_flat.shape[0])
        if batch_size == 1 and num_cams == num_candidates:
            return pts_world_flat.view(1, num_candidates, -1, 3)
        if num_cams == (batch_size * num_candidates):
            return pts_world_flat.view(batch_size, num_candidates, -1, 3)
        raise ValueError(
            "p3d_cameras batch size must be N (when B=1) or B*N; "
            f"got {num_cams} for B={batch_size}, N={num_candidates}.",
        )

    def sample_field(
        self,
        field: Tensor,
        *,
        points_world: Tensor,
        t_world_voxel: PoseTW,
        voxel_extent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample voxel-aligned features at world-space points.

        Args:
            field: ``Tensor["B C D H W", float32]`` voxel field.
            points_world: ``Tensor["B N K 3", float32]`` world-space points.
            t_world_voxel: ``PoseTW["B 12"]`` world<-voxel transform.
            voxel_extent: ``Tensor["B 6", float32]`` voxel bounds in voxel frame.

        Returns:
            Tuple of:
                - tokens: ``Tensor["B N K C", float32]``
                - valid: ``Tensor["B N K", bool]``
        """
        return _sample_voxel_field(
            field,
            points_world=points_world,
            t_world_voxel=t_world_voxel,
            voxel_extent=voxel_extent,
        )

    def pool_tokens(self, *, tokens: Tensor, valid: Tensor) -> Tensor:
        """Mean-pool candidate-local frustum samples.

        Args:
            tokens: ``Tensor["B N K C", float32]`` sampled voxel features.
            valid: ``Tensor["B N K", bool]`` in-bounds mask.

        Returns:
            ``Tensor["B N C", float32]`` masked mean over K samples.
        """
        if tokens.ndim != 4:
            raise ValueError(
                f"Expected tokens shape (B,N,K,C), got {tuple(tokens.shape)}.",
            )
        if valid.shape != tokens.shape[:3]:
            raise ValueError(
                f"Expected valid shape {tuple(tokens.shape[:3])}, got {tuple(valid.shape)}.",
            )

        mask = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        return (tokens * mask).sum(dim=-2) / denom


class VinFeatureAssembler:
    """Combine pose, global, voxel, and local tokens into per-candidate features."""

    def __init__(
        self,
        config: VinPipelineConfig,
        *,
        pose_dim: int,
        field_dim: int,
    ) -> None:
        self.config = config
        self.pose_dim = pose_dim
        self.field_dim = field_dim
        self.out_dim = pose_dim + field_dim
        if self.config.use_global_pool:
            self.out_dim += field_dim
        if self.config.use_voxel_pose_encoding:
            self.out_dim += pose_dim

    @staticmethod
    def _pool_global(field: Tensor) -> Tensor:
        """Mean-pool global context from a voxel field."""
        return field.mean(dim=(-3, -2, -1))

    def assemble(
        self,
        *,
        pose_enc: Tensor,
        field: Tensor,
        local_feat: Tensor,
        voxel_pose_enc: Tensor | None,
        batch_size: int,
        num_candidates: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Assemble per-candidate features and return global token if used.

        Args:
            pose_enc: ``Tensor["B N E_pose", float32]`` pose embeddings.
            field: ``Tensor["B C D H W", float32]`` voxel field.
            local_feat: ``Tensor["B N C", float32]`` pooled frustum features.
            voxel_pose_enc: Optional ``Tensor["B E_pose", float32]`` voxel pose embedding.
            batch_size: Batch size ``B``.
            num_candidates: Candidate count ``N``.

        Returns:
            Tuple of:
                - feats: ``Tensor["B N F", float32]`` assembled features.
                - global_feat: ``Tensor["B N C", float32]`` or ``None``.
        """
        parts: list[Tensor] = [pose_enc.to(dtype=field.dtype)]
        global_feat: Tensor | None = None
        if self.config.use_global_pool:
            global_feat = (
                self._pool_global(field)
                .unsqueeze(1)
                .expand(batch_size, num_candidates, -1)
            )
            parts.append(global_feat)
        if self.config.use_voxel_pose_encoding and voxel_pose_enc is not None:
            voxel_feat = voxel_pose_enc.to(dtype=field.dtype).unsqueeze(1)
            parts.append(voxel_feat.expand(batch_size, num_candidates, -1))

        parts.append(local_feat.to(dtype=field.dtype))
        feats = torch.cat(parts, dim=-1)
        return feats, global_feat


class VinPipelineConfig(BaseConfig["VinPipeline"]):
    """Single configuration shared by all VIN pipeline components."""

    target: type[VinPipeline] = Field(default_factory=lambda: VinPipeline, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(input_dim=8),
    )
    """Learnable Fourier Features pose encoding configuration (shell descriptor)."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: list[str] = Field(
        default_factory=lambda: ["occ_pr"],
        min_length=1,
    )
    """Ordered channels used to build the low-dimensional scene field."""

    occ_input_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Threshold used when deriving fallback free-space evidence from `occ_input`."""

    counts_norm_mode: Literal["log1p", "linear"] = "log1p"
    """How to normalize voxel `counts` into [0, 1]."""

    occ_pr_is_logits: bool = False
    """Whether `occ_pr` is logits (apply sigmoid) rather than a probability volume."""

    field_dim: int = Field(default=16, gt=0)
    """Channel dimension d0 of the compressed scene field."""

    field_gn_groups: int = Field(default=4, gt=0)
    """Requested GroupNorm groups for the field projection (clamped to a divisor of `field_dim`)."""

    frustum_grid_size: int = Field(default=4, gt=0)
    """Grid size on the image plane for candidate frustum sampling (grid_size² directions)."""

    frustum_depths_m: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0],
        min_length=1,
    )
    """Depth values (metres) along each frustum direction."""

    use_global_pool: bool = True
    """Whether to concatenate the global mean-pooled embedding to per-candidate features."""

    use_voxel_pose_encoding: bool = True
    """Whether to append a SH-encoded voxel-grid pose (voxel/T_world_voxel) in the reference frame."""

    candidate_min_valid_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    """Minimum fraction of valid frustum samples required to keep a candidate."""

    @field_validator("scene_field_channels")
    @classmethod
    def _validate_scene_field_channels(cls, value: list[str]) -> list[str]:
        """Validate requested scene-field channels."""
        allowed = {
            "occ_pr",
            "occ_input",
            "counts_norm",
            "observed",
            "unknown",
            "new_surface_prior",
            "free_input",
            "cent_pr",
        }
        unknown = [name for name in value if name not in allowed]
        if unknown:
            raise ValueError(f"Unknown/unsupported scene_field_channels: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError("scene_field_channels must not contain duplicates.")
        return value

    @field_validator("frustum_depths_m")
    @classmethod
    def _validate_frustum_depths_m(cls, value: list[float]) -> list[float]:
        """Validate frustum depths."""
        bad = [d for d in value if (not math.isfinite(d)) or d <= 0.0]
        if bad:
            raise ValueError(
                f"frustum_depths_m must contain finite values > 0, got {bad}",
            )
        return value

    @field_validator("pose_encoder_lff")
    @classmethod
    def _validate_pose_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        """Ensure the LFF input dimensionality matches the pose vector definition."""
        if value.input_dim != 8:
            raise ValueError(
                "pose_encoder_lff.input_dim must be 8 for [u, f, r, s] pose vectors.",
            )
        return value


class VinPipeline(nn.Module):
    """Modular VIN implementation for side-by-side validation with VinModel."""

    def __init__(self, config: VinPipelineConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = self.config.backbone.setup_target()
        self.pose_encoder = VinPoseEncoder(self.config)
        self.scene_field = VinSceneFieldBuilder(self.config)
        self.frustum_sampler = VinFrustumSampler(self.config)
        self.feature_assembler = VinFeatureAssembler(
            self.config,
            pose_dim=self.pose_encoder.out_dim,
            field_dim=self.scene_field.field_dim,
        )
        self.head = self.config.head.setup_target(in_dim=self.feature_assembler.out_dim)
        self.to(self.backbone.device)

    def _forward_impl(
        self,
        efm: dict[str, Any],
        *,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        return_debug: bool,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinForwardDiagnostics | None]:
        """Shared forward pass for prediction and debug outputs."""
        if backbone_out is None:
            backbone_out = self.backbone.forward(efm)
        device = backbone_out.voxel_extent.device

        pose_world_cam = _ensure_candidate_batch(candidate_poses_world_cam).to(
            device=device,
        )
        batch_size = int(pose_world_cam.shape[0])
        num_candidates = int(pose_world_cam.shape[1])

        pose_world_rig_ref = _ensure_pose_batch(
            reference_pose_world_rig.to(device=device),
            batch_size=batch_size,
            name="reference_pose_world_rig",
        )

        # Candidate pose in reference rig frame.
        pose_rig_cam = pose_world_rig_ref.inverse()[:, None] @ pose_world_cam
        pose_shell = self.pose_encoder.encode(pose_rig_cam)

        # Voxel pose in reference rig frame.
        t_world_voxel = _ensure_pose_batch(
            backbone_out.t_world_voxel,
            batch_size=batch_size,
            name="voxel/T_world_voxel",
        )
        pose_rig_voxel = pose_world_rig_ref.inverse() @ t_world_voxel
        voxel_shell = self.pose_encoder.encode(pose_rig_voxel)

        # Scene field.
        scene_field = self.scene_field(backbone_out)
        field = scene_field.field.to(device=device)

        # Frustum query.
        points_world = self.frustum_sampler.build_points_world(
            pose_world_cam,
            p3d_cameras=p3d_cameras,
        )
        tokens, token_valid = self.frustum_sampler.sample_field(
            field,
            points_world=points_world,
            t_world_voxel=backbone_out.t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        valid_frac = token_valid.float().mean(dim=-1)
        local_feat = self.frustum_sampler.pool_tokens(tokens=tokens, valid=token_valid)

        # Feature assembly.
        feats, global_feat = self.feature_assembler.assemble(
            pose_enc=pose_shell.pose_enc,
            field=field,
            local_feat=local_feat,
            voxel_pose_enc=voxel_shell.pose_enc
            if self.config.use_voxel_pose_encoding
            else None,
            batch_size=batch_size,
            num_candidates=num_candidates,
        )

        candidate_valid = _candidate_valid_from_token(
            token_valid,
            min_valid_frac=self.config.candidate_min_valid_frac,
        )
        feats = feats * candidate_valid.to(dtype=feats.dtype).unsqueeze(-1)

        logits = self.head(feats.reshape(batch_size * num_candidates, -1)).reshape(
            batch_size,
            num_candidates,
            -1,
        )
        prob = coral_logits_to_prob(logits)
        expected, expected_norm = coral_expected_from_logits(logits)

        pred = VinPrediction(
            logits=logits,
            prob=prob,
            expected=expected,
            expected_normalized=expected_norm,
            candidate_valid=candidate_valid,
            valid_frac=valid_frac,
        )

        if not return_debug:
            return pred, None

        debug = VinForwardDiagnostics(
            backbone_out=backbone_out,
            candidate_center_rig_m=pose_shell.center_m,
            candidate_radius_m=pose_shell.radius_m,
            candidate_center_dir_rig=pose_shell.center_dir,
            candidate_forward_dir_rig=pose_shell.forward_dir,
            view_alignment=pose_shell.view_alignment,
            pose_enc=pose_shell.pose_enc,
            pose_vec=None,
            voxel_center_rig_m=voxel_shell.center_m,
            voxel_radius_m=voxel_shell.radius_m,
            voxel_center_dir_rig=voxel_shell.center_dir,
            voxel_forward_dir_rig=voxel_shell.forward_dir,
            voxel_view_alignment=voxel_shell.view_alignment,
            voxel_pose_enc=voxel_shell.pose_enc
            if self.config.use_voxel_pose_encoding
            else None,
            voxel_pose_vec=None,
            field_in=scene_field.field_in,
            field=field,
            global_feat=global_feat,
            local_feat=local_feat,
            tokens=tokens,
            token_valid=token_valid,
            candidate_valid=candidate_valid,
            valid_frac=valid_frac.unsqueeze(-1),
            feats=feats,
        )
        return pred, debug

    def forward(
        self,
        efm: dict[str, Any],
        *,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> VinPrediction:
        """Score candidate poses for one snippet (no diagnostics).

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: ``PoseTW["N 12"]`` or ``PoseTW["B N 12"]`` candidates.
            reference_pose_world_rig: ``PoseTW["12"]`` or ``PoseTW["B 12"]`` reference rig pose.
            p3d_cameras: PyTorch3D cameras aligned with candidates.
            backbone_out: Optional precomputed :class:`EvlBackboneOutput`.

        Returns:
            :class:`VinPrediction` with CORAL logits and derived scores.
        """
        pred, _ = self._forward_impl(
            efm,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            p3d_cameras=p3d_cameras,
            return_debug=False,
            backbone_out=backbone_out,
        )
        return pred

    def forward_with_debug(
        self,
        efm: dict[str, Any],
        *,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinForwardDiagnostics]:
        """Run VIN forward pass and return intermediate tensors.

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: ``PoseTW["N 12"]`` or ``PoseTW["B N 12"]`` candidates.
            reference_pose_world_rig: ``PoseTW["12"]`` or ``PoseTW["B 12"]`` reference rig pose.
            p3d_cameras: PyTorch3D cameras aligned with candidates.
            backbone_out: Optional precomputed :class:`EvlBackboneOutput`.

        Returns:
            Tuple of (:class:`VinPrediction`, :class:`VinForwardDiagnostics`).
        """
        pred, debug = self._forward_impl(
            efm,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            p3d_cameras=p3d_cameras,
            return_debug=True,
            backbone_out=backbone_out,
        )
        if debug is None:
            raise RuntimeError("Expected VinForwardDiagnostics when return_debug=True.")
        return pred, debug

"""Shared helpers and config surfaces for experimental VIN model variants.

This module centralizes the helper functions and small config/class surfaces
that are shared by both experimental VIN model implementations:

- frustum construction and voxel-field sampling helpers,
- scene-field construction from EVL backbone outputs,
- the shared CORAL scoring head,
- the shared scorer-head config, and
- the two variant-specific model configs under distinct concrete names.

The concrete model modules re-export these configs back under their historical
``VinModelConfig`` / ``VinScorerHeadConfig`` names so external imports stay
stable while the duplicate definitions disappear.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from efm3d.aria.pose import PoseTW
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field, field_validator, model_validator
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)
from torch import Tensor, nn

from ...rri_metrics.coral import CoralLayer
from ...utils import BaseConfig
from ..backbone_evl import EvlBackboneConfig
from .pose_encoding import LearnableFourierFeaturesConfig
from .spherical_encoding import ShellShPoseEncoderConfig
from .types import EvlBackboneOutput


def _largest_divisor_leq(n: int, max_divisor: int) -> int:
    """Return the largest divisor of ``n`` that is less than ``max_divisor``."""
    groups = min(max_divisor, n)
    while groups > 1 and (n % groups) != 0:
        groups -= 1
    return max(1, groups)


def _build_frustum_points_world_p3d(
    cameras: PerspectiveCameras,
    *,
    grid_size: int,
    depths_m: list[float],
) -> Tensor:
    """Unproject a small frustum grid into world points at fixed metric depths."""
    num_cams = int(cameras.R.shape[0])
    device = cameras.R.device

    image_size = cameras.image_size.to(device=device, dtype=torch.float32)
    principal_point = cameras.principal_point.to(device=device, dtype=torch.float32)
    if image_size.shape[0] == 1 and num_cams > 1:
        image_size = image_size.expand(num_cams, -1)
    if principal_point.shape[0] == 1 and num_cams > 1:
        principal_point = principal_point.expand(num_cams, -1)

    h = image_size[:, 0]
    w = image_size[:, 1]
    scale = torch.minimum(h, w)

    half = 0.95 * 0.5 * scale
    half_x = torch.minimum(half, torch.minimum(principal_point[:, 0] - 0.5, (w - 0.5) - principal_point[:, 0]))
    half_y = torch.minimum(half, torch.minimum(principal_point[:, 1] - 0.5, (h - 0.5) - principal_point[:, 1]))
    half_x = torch.clamp(half_x, min=0.0)
    half_y = torch.clamp(half_y, min=0.0)

    u_min = principal_point[:, 0] - half_x
    u_max = principal_point[:, 0] + half_x
    v_min = principal_point[:, 1] - half_y
    v_max = principal_point[:, 1] + half_y

    t = torch.linspace(0.0, 1.0, steps=grid_size, device=device, dtype=torch.float32)
    us = u_min[:, None] + (u_max - u_min)[:, None] * t[None, :]
    vs = v_min[:, None] + (v_max - v_min)[:, None] * t[None, :]

    uu = us[:, None, :].expand(num_cams, grid_size, grid_size)
    vv = vs[:, :, None].expand(num_cams, grid_size, grid_size)

    u = uu.reshape(num_cams, -1)
    v = vv.reshape(num_cams, -1)

    x_ndc = -(u - w[:, None] * 0.5) * (2.0 / scale[:, None])
    y_ndc = -(v - h[:, None] * 0.5) * (2.0 / scale[:, None])

    depths = torch.tensor(depths_m, device=device, dtype=torch.float32)
    num_depths = int(depths.shape[0])
    num_rays = int(x_ndc.shape[1])

    x_ndc = x_ndc[:, None, :].expand(num_cams, num_depths, num_rays)
    y_ndc = y_ndc[:, None, :].expand(num_cams, num_depths, num_rays)
    z = depths.view(1, num_depths, 1).expand(num_cams, num_depths, num_rays)

    xy_depth = torch.stack([x_ndc, y_ndc, z], dim=-1).reshape(num_cams, -1, 3)
    return cameras.unproject_points(xy_depth, world_coordinates=True, from_ndc=True)


def _build_scene_field(
    out: EvlBackboneOutput,
    *,
    use_channels: list[str],
    occ_input_threshold: float,
    counts_norm_mode: Literal["log1p", "linear"],
    occ_pr_is_logits: bool,
) -> Tensor:
    """Build a compact voxel-aligned scene field from EVL head/evidence tensors."""

    def _require(name: str) -> Tensor:
        value = getattr(out, name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Missing backbone output '{name}'. Ensure EvlBackboneConfig.features_mode includes 'heads'."
            )
        return value

    parts: dict[str, Tensor] = {}

    if "occ_pr" in use_channels or "new_surface_prior" in use_channels:
        occ_pr = _require("occ_pr").to(dtype=torch.float32)
        parts["occ_pr"] = torch.sigmoid(occ_pr) if occ_pr_is_logits else occ_pr

    if "occ_input" in use_channels or "free_input" in use_channels:
        parts["occ_input"] = _require("occ_input").to(dtype=torch.float32)

    if "cent_pr" in use_channels:
        parts["cent_pr"] = _require("cent_pr").to(dtype=torch.float32)

    if "free_input" in use_channels:
        if isinstance(out.free_input, torch.Tensor):
            parts["free_input"] = out.free_input.to(dtype=torch.float32)
        else:
            counts = _require("counts")
            observed = (counts > 0).to(dtype=torch.float32).unsqueeze(1)
            occ_evidence = (parts["occ_input"] > occ_input_threshold).to(dtype=torch.float32)
            parts["free_input"] = observed * (1.0 - occ_evidence)

    if (
        "counts_norm" in use_channels
        or "observed" in use_channels
        or "unknown" in use_channels
        or "new_surface_prior" in use_channels
    ):
        counts = _require("counts").to(dtype=torch.float32)

        observed = (counts > 0).to(dtype=torch.float32).unsqueeze(1)
        parts["observed"] = observed
        parts["unknown"] = 1.0 - observed

        max_counts = counts.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        if counts_norm_mode == "log1p":
            parts["counts_norm"] = torch.log1p(counts).unsqueeze(1) / torch.log1p(max_counts).unsqueeze(1)
        else:
            parts["counts_norm"] = (counts / max_counts).unsqueeze(1)

    if "new_surface_prior" in use_channels:
        parts["new_surface_prior"] = parts["unknown"] * parts["occ_pr"]

    stacked = [parts[name] for name in use_channels]
    return torch.cat(stacked, dim=1)


def _sample_voxel_field(
    field: Tensor,
    points_world: Tensor,
    *,
    t_world_voxel: PoseTW,
    voxel_extent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sample a voxel field at world-space query points."""
    if field.ndim != 5:
        raise ValueError(f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.")
    if points_world.ndim != 4:
        raise ValueError(f"Expected points_world shape (B,N,K,3), got {tuple(points_world.shape)}.")
    if int(points_world.shape[-1]) != 3:
        raise ValueError(f"Expected points_world[..., 3], got {tuple(points_world.shape)}.")

    batch_size, field_channels, _grid_d, _grid_h, _grid_w = field.shape
    _, num_candidates, num_points, _ = points_world.shape

    t_world_voxel_b = t_world_voxel.to(device=field.device)
    if t_world_voxel_b.ndim == 1:
        t_world_voxel_b = PoseTW(t_world_voxel_b._data.view(1, 12).expand(batch_size, 12))
    elif int(t_world_voxel_b.shape[0]) != batch_size:
        if int(t_world_voxel_b.shape[0]) == 1:
            t_world_voxel_b = PoseTW(t_world_voxel_b._data.expand(batch_size, 12))
        else:
            raise ValueError("t_world_voxel must have batch size 1 or match field batch size.")

    vox_extent = voxel_extent.to(device=field.device, dtype=torch.float32)
    if vox_extent.ndim == 1:
        vox_extent = vox_extent.view(1, 6).expand(batch_size, 6)
    if vox_extent.shape != (batch_size, 6):
        raise ValueError(f"Expected voxel_extent shape (B,6), got {tuple(vox_extent.shape)}.")

    world_points_flat = points_world.to(device=field.device, dtype=field.dtype).reshape(
        batch_size, num_candidates * num_points, 3
    )
    points_voxel = t_world_voxel_b.inverse() * world_points_flat
    points_voxel = points_voxel.to(dtype=torch.float32)
    pts_vox_id, valid_extent = pc_to_vox(points_voxel, vox_extent, field.shape[-3:])
    pts_vox_id = torch.nan_to_num(pts_vox_id, nan=0.0, posinf=0.0, neginf=0.0)

    sampled, valid_grid = sample_voxels(field, pts_vox_id, differentiable=False)
    valid = (valid_extent & valid_grid).reshape(batch_size, num_candidates, num_points)
    tokens = sampled.transpose(1, 2).reshape(batch_size, num_candidates, num_points, field_channels)
    return tokens, valid


def _candidate_valid_from_token(token_valid: Tensor, *, min_valid_frac: float) -> Tensor:
    """Convert per-token validity into a per-candidate mask."""
    if token_valid.ndim < 1:
        raise ValueError(f"Expected token_valid with ndim>=1, got {tuple(token_valid.shape)}.")
    valid_frac = token_valid.float().mean(dim=-1)
    return valid_frac >= min_valid_frac


class VinScorerHead(nn.Module):
    """Candidate scoring head producing CORAL ordinal logits."""

    def __init__(self, config: "VinScorerHeadConfig", *, in_dim: int | None = None) -> None:
        """Initialize the shared ordinal scorer head."""
        super().__init__()
        self.config = config

        act: nn.Module
        match self.config.activation:
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()

        hidden_dim = self.config.hidden_dim
        layers: list[nn.Module] = []
        if in_dim is None:
            layers.append(nn.LazyLinear(hidden_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)
        if self.config.dropout > 0:
            layers.append(nn.Dropout(p=self.config.dropout))

        for _ in range(self.config.num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if self.config.dropout > 0:
                layers.append(nn.Dropout(p=self.config.dropout))

        self.mlp = nn.Sequential(*layers)
        self.coral = CoralLayer(in_dim=hidden_dim, num_classes=self.config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute CORAL logits from per-candidate features."""
        return self.coral(self.mlp(x))


class VinScorerHeadConfig(BaseConfig[VinScorerHead]):
    """Configuration for the shared experimental VIN CORAL scorer head."""

    @property
    def target(self) -> type[VinScorerHead]:
        """Return the scorer-head factory target."""
        return VinScorerHead

    hidden_dim: int = Field(default=128, gt=0)
    """Hidden dimension for MLP layers."""

    num_layers: int = Field(default=1, ge=1)
    """Number of MLP layers before the CORAL layer."""

    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Dropout probability in the MLP."""

    num_classes: int = Field(default=15, ge=2)
    """Number of ordinal bins."""

    activation: Literal["gelu", "relu"] = "gelu"
    """Activation function used in the scorer MLP."""

    def setup_target(self, *, in_dim: int | None = None) -> VinScorerHead:  # type: ignore[override]
        """Instantiate the scorer head with an optional explicit input size."""
        return self.target(self, in_dim=in_dim)


class LffVinModelConfig(BaseConfig["VinModel"]):
    """Configuration for the LFF-based experimental VIN model."""

    @property
    def target(self) -> type[nn.Module]:
        """Return the LFF VIN model factory target."""
        from .model import VinModel

        return VinModel

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Optional frozen EVL backbone configuration."""

    pose_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(input_dim=9),
        validation_alias="pose_encoder_sh",
    )
    """Learnable Fourier Features pose encoding configuration."""

    pose_encoding_mode: Literal["shell_lff", "t_r6d_lff"] = "t_r6d_lff"
    """Pose vector used for LFF: shell descriptor or translation plus rotation-6D."""

    pose_scale_init: tuple[float, float] = (1.0, 1.0)
    """Initial per-group scale for translation and rotation-6D inputs."""

    pose_scale_learnable: bool = True
    """Whether pose scaling parameters are learned."""

    pose_scale_eps: float = Field(default=1e-6, gt=0.0)
    """Numerical floor for pose scaling."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: list[
        Literal[
            "occ_pr",
            "occ_input",
            "counts_norm",
            "observed",
            "unknown",
            "new_surface_prior",
            "free_input",
            "cent_pr",
        ]
    ] = Field(default_factory=lambda: ["occ_pr"], min_length=1)
    """Ordered channels used to build the low-dimensional scene field."""

    occ_input_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Threshold used when deriving fallback free-space evidence from ``occ_input``."""

    counts_norm_mode: Literal["log1p", "linear"] = "log1p"
    """How to normalize voxel ``counts`` into ``[0, 1]``."""

    occ_pr_is_logits: bool = False
    """Whether ``occ_pr`` is logits rather than a probability volume."""

    field_dim: int = Field(default=16, gt=0)
    """Channel dimension of the compressed scene field."""

    field_gn_groups: int = Field(default=4, gt=0)
    """Requested GroupNorm groups for the field projection."""

    frustum_grid_size: int = Field(default=4, gt=0)
    """Grid size on the image plane for candidate frustum sampling."""

    frustum_depths_m: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0], min_length=1)
    """Depth values along each frustum direction."""

    use_global_pool: bool = True
    """Whether to concatenate a global context token to per-candidate features."""

    global_pool_mode: Literal["mean", "mean_max", "attn"] = "attn"
    """Global pooling mode."""

    global_pool_grid_size: int = Field(default=8, gt=0)
    """Target grid size for attention pooling."""

    global_pool_dim: int | None = None
    """Attention embedding dimension."""

    global_pool_heads: int = Field(default=4, gt=0)
    """Number of attention heads for pose-conditioned pooling."""

    global_pool_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Dropout rate for attention pooling."""

    use_voxel_pose_encoding: bool = True
    """Whether to append an encoded voxel-grid pose token."""

    use_unknown_token: bool = True
    """Whether to replace invalid frustum samples with a learned unknown token."""

    use_valid_frac_feature: bool = True
    """Whether to append the valid-frustum fraction as scalar features."""

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

    @field_validator("pose_encoder_lff")
    @classmethod
    def _validate_pose_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        """Ensure the LFF input dimensionality matches the pose vector definition."""
        if value.input_dim not in (8, 9):
            raise ValueError("pose_encoder_lff.input_dim must be 8 (shell) or 9 (t+R6d).")
        return value

    @field_validator("global_pool_dim")
    @classmethod
    def _validate_global_pool_dim(cls, value: int | None) -> int | None:
        """Validate attention embedding dimension when provided."""
        if value is not None and value <= 0:
            raise ValueError("global_pool_dim must be > 0 when provided.")
        return value

    @field_validator("pose_scale_init")
    @classmethod
    def _validate_pose_scale_init(
        cls,
        value: tuple[float, float],
    ) -> tuple[float, float]:
        """Validate the translation and rotation pose scale tuple."""
        if len(value) != 2:
            raise ValueError("pose_scale_init must be a (translation, rotation) tuple.")
        return value

    @model_validator(mode="after")
    def _validate_pose_encoding_mode(self) -> "LffVinModelConfig":
        """Ensure the pose-encoder input matches the requested encoding mode."""
        expected = 8 if self.pose_encoding_mode == "shell_lff" else 9
        if self.pose_encoder_lff.input_dim != expected:
            raise ValueError(
                "pose_encoder_lff.input_dim must match pose_encoding_mode "
                f"({self.pose_encoding_mode} expects {expected})."
            )
        return self


class ShellVinModelConfig(BaseConfig["VinModel"]):
    """Configuration for the shell-harmonics experimental VIN model."""

    @property
    def target(self) -> type[nn.Module]:
        """Return the shell VIN model factory target."""
        from .model_v1_SH import VinModel

        return VinModel

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoder_sh: ShellShPoseEncoderConfig = Field(default_factory=ShellShPoseEncoderConfig)
    """Spherical harmonics pose encoding configuration."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: list[
        Literal[
            "occ_pr",
            "occ_input",
            "counts_norm",
            "observed",
            "unknown",
            "new_surface_prior",
            "free_input",
        ]
    ] = Field(default_factory=lambda: ["occ_pr", "occ_input", "counts_norm"], min_length=1)
    """Ordered channels used to build the low-dimensional scene field."""

    occ_input_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Threshold used when deriving fallback free-space evidence from ``occ_input``."""

    counts_norm_mode: Literal["log1p", "linear"] = "log1p"
    """How to normalize voxel ``counts`` into ``[0, 1]``."""

    occ_pr_is_logits: bool = False
    """Whether ``occ_pr`` is logits rather than a probability volume."""

    field_dim: int = Field(default=16, gt=0)
    """Channel dimension of the compressed scene field."""

    field_gn_groups: int = Field(default=4, gt=0)
    """Requested GroupNorm groups for the field projection."""

    frustum_grid_size: int = Field(default=4, gt=0)
    """Grid size on the image plane for candidate frustum sampling."""

    frustum_depths_m: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0], min_length=1)
    """Depth values along each frustum direction."""

    use_global_pool: bool = True
    """Whether to concatenate the global pooled embedding to per-candidate features."""

    use_voxel_pose_encoding: bool = True
    """Whether to append an encoded voxel-grid pose token."""

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
        """Validate that frustum depths are finite and strictly positive."""
        bad = [depth for depth in value if (not math.isfinite(depth)) or depth <= 0.0]
        if bad:
            raise ValueError(f"frustum_depths_m must contain finite values > 0, got {bad}")
        return value


build_frustum_points_world_p3d = _build_frustum_points_world_p3d
build_scene_field = _build_scene_field
candidate_valid_from_token = _candidate_valid_from_token
largest_divisor_leq = _largest_divisor_leq
sample_voxel_field = _sample_voxel_field

__all__ = [
    "LffVinModelConfig",
    "ShellVinModelConfig",
    "VinScorerHead",
    "VinScorerHeadConfig",
    "build_frustum_points_world_p3d",
    "build_scene_field",
    "candidate_valid_from_token",
    "largest_divisor_leq",
    "sample_voxel_field",
]

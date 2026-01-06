"""VIN v2 (simplified) for RRI prediction with fixed, transparent components.

This module provides a **reduced-configuration** VIN variant that keeps the
most promising architectural pieces while removing mode switches:

1) **Pose encoding (configurable).**
   We express each candidate pose in the reference rig frame,

       T_rig_ref_cam = T_world_rig_ref^{-1} * T_world_cam,

   then encode it with a configurable pose encoder. The default is translation +
   rotation-6D passed through LFF (with learned per-group scaling). Optional
   shell-based LFF/SH encoders are available for experimentation.

2) **Scene field (fixed channels, no hard thresholds).**
   We build a compact voxel field with the most RRI-relevant channels:

       occ_pr, cent_pr, counts_norm, occ_input, free_input, new_surface_prior,

   where ``counts_norm`` is log1p-normalized coverage, ``unknown`` is treated as
   ``1 - counts_norm`` (soft), and ``new_surface_prior = unknown * occ_pr``.
   The field is projected via ``1x1x1 Conv3d + GroupNorm + GELU``.

3) **Global context (pose-conditioned attention).**
   A coarse voxel grid is pooled and attended by the pose embeddings. Keys are
   augmented with an LFF positional encoding of XYZ voxel centers derived from
   ``voxel/pts_world`` after mapping those points into the **reference rig frame**.

4) **CORAL head.**
   We concatenate pose and global tokens and score with an MLP + CORAL ordinal
   head.

Frame-consistency note:
Candidate generation applies ``rotate_yaw_cw90`` (a local +Z roll) to the
reference/candidate poses for UI alignment. EVL backbone outputs do **not**
use this convention. ``VinModelV2`` therefore **undoes** this rotation
before computing pose features.
"""
# TODO: try using other features - e.g. for example use dinov2 features of the input rgbs and transform feature grid into candidate frame (as done in the original vin nbv pape docs/contents/literature/vin_nbv.qmd)
# TODO(not-now): try add learnable parameters to shift the binning edges of coral layer.
# TODO: How can we make use of the original semi-dense pointcloud from the efm_snippet rather than relying only on the voxel grid?

# TODO: reimplement frustum based sampling - we can use this to do sparse pooling of the voxel field. however, here we need to consider that some candidates may not even see any voxels as the voxel extends are quite limited (4x4x4m) symmetric around the last rig pose.
# TODO(not now): use voxel reference frame as our reference around which we generate candidate poses in candidate_generation.py!

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from efm3d.aria.aria_constants import (
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_INV_DIST_STD,
    ARIA_POINTS_TIME_NS,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
)
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)
from torch import Tensor, nn
from torch.nn import functional as functional

from ..data.efm_views import EfmPointsView
from ..rri_metrics.coral import (
    CoralLayer,
    coral_expected_from_logits,
    coral_logits_to_prob,
)
from ..utils import BaseConfig, Optimizable, optimizable_field, rotate_yaw_cw90
from .backbone_evl import EvlBackboneConfig
from .model import _largest_divisor_leq, _sample_voxel_field
from .pose_encoders import PoseEncoder, PoseEncoderConfig, R6dLffPoseEncoderConfig
from .pose_encoding import LearnableFourierFeaturesConfig
from .types import EvlBackboneOutput, VinPrediction

if TYPE_CHECKING:
    from efm3d.aria.pose import PoseTW as PoseTWT

    from oracle_rri.lightning.lit_datamodule import VinOracleBatch

    from .pose_encoding import LearnableFourierFeatures


@dataclass(slots=True)
class VinV2ForwardDiagnostics:
    """Minimal diagnostics for VIN v2 (no frustum or shell encodings)."""

    backbone_out: EvlBackboneOutput
    """EVL backbone outputs used to build the scene field."""

    candidate_center_rig_m: Tensor
    """``Tensor["B N 3", float32]`` Candidate centers in the reference rig frame."""

    pose_enc: Tensor
    """``Tensor["B N E_pose", float32]`` Pose encoder output."""

    pose_vec: Tensor
    """``Tensor["B N D_pose", float32]`` Pose vector fed into the pose encoder."""

    field_in: Tensor
    """``Tensor["B C_in D H W", float32]`` Raw scene field before projection."""

    field: Tensor
    """``Tensor["B C_out D H W", float32]`` Projected scene field."""

    global_feat: Tensor
    """``Tensor["B N C_global", float32]`` Pose-conditioned global features."""

    feats: Tensor
    """``Tensor["B N F", float32]`` Concatenated VIN features."""

    candidate_valid: Tensor
    """``Tensor["B N", bool]`` Candidate validity mask (finite pose + in-bounds center)."""

    voxel_valid_frac: Tensor | None = None
    """``Tensor["B N", float32]`` Per-candidate voxel coverage proxy (if computed)."""


FIELD_CHANNELS_V2: tuple[str, ...] = (
    "occ_pr",
    "occ_input",
    "counts_norm",
    "observed",
    "unknown",
    "new_surface_prior",
    "free_input",
    "cent_pr",
)


class _AttrDict(dict[str, Any]):
    """Dict that exposes keys as attributes (minimal EasyDict fallback)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _extract_tensor(output: Any) -> Tensor:
    """Best-effort extractor for tensors returned by external point encoders."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("feat", "features", "logits", "pred", "out"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("Point encoder output did not contain a tensor.")


def _load_openpoints_cfg(path: Path) -> Any:
    """Load an OpenPoints YAML config, returning an attribute-accessible object."""

    def _to_attr(obj: Any) -> Any:
        if isinstance(obj, dict):
            return _AttrDict({k: _to_attr(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_attr(v) for v in obj]
        return obj

    try:
        from openpoints.utils import EasyConfig  # type: ignore[import-not-found]
    except Exception:
        easy_config_cls = None
    else:
        easy_config_cls = EasyConfig

    if easy_config_cls is not None:
        cfg = easy_config_cls()
        cfg.load(str(path), recursive=True)
        return cfg

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "PointNeXt-S config loading requires either openpoints (EasyConfig) or pyyaml to parse the YAML file.",
        ) from exc

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _to_attr(data)


def _strip_state_prefix(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    if not prefix:
        return state
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()}


class PointNeXtSEncoderConfig(BaseConfig["PointNeXtSEncoder"]):
    """Configuration for the optional PointNeXt-S semidense encoder."""

    target: type["PointNeXtSEncoder"] = Field(default_factory=lambda: PointNeXtSEncoder, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    cfg_path: Path
    """Path to the OpenPoints PointNeXt-S YAML config (e.g. model zoo config)."""

    checkpoint_path: Path | None = None
    """Optional pretrained checkpoint (OpenPoints format)."""

    out_dim: int = Field(default=128, gt=0)
    """Output embedding dimension produced for the semidense point cloud."""

    max_points: int = Field(default=3000, gt=0)
    """Subsample semidense points to this count before encoding."""

    freeze: bool = True
    """Whether to freeze PointNeXt-S weights during training."""

    strict_load: bool = False
    """Whether to enforce strict checkpoint loading."""

    @field_validator("cfg_path", "checkpoint_path", mode="before")
    @classmethod
    def _coerce_paths(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value)


class PointNeXtSEncoder(nn.Module):
    """Optional PointNeXt-S adapter for semidense point cloud features."""

    def __init__(self, config: PointNeXtSEncoderConfig) -> None:
        super().__init__()
        self.config = config

        try:
            from openpoints.models import build_model_from_cfg  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "PointNeXt-S requires the openpoints package. Install it and provide a valid cfg_path/checkpoint.",
            ) from exc

        cfg_path = Path(self.config.cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"PointNeXt-S cfg_path does not exist: {cfg_path}")
        cfg = _load_openpoints_cfg(cfg_path)
        model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
        self.model = build_model_from_cfg(model_cfg)
        self.model.eval()

        if self.config.checkpoint_path is not None:
            ckpt_path = Path(self.config.checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"PointNeXt-S checkpoint_path does not exist: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = _strip_state_prefix(state, "module.")
                self.model.load_state_dict(state, strict=self.config.strict_load)

        if self.config.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

        with torch.no_grad():
            dummy = torch.zeros((1, 32, 3), dtype=torch.float32)
            raw = _extract_tensor(self._forward_raw(dummy))
            raw_dim = int(raw.shape[-1])

        self.out_dim = int(self.config.out_dim)
        self.proj = nn.Identity() if raw_dim == self.out_dim else nn.Linear(raw_dim, self.out_dim)

    def train(self, mode: bool = True) -> "PointNeXtSEncoder":
        super().train(mode)
        if self.config.freeze:
            self.model.eval()
        return self

    def _forward_raw(self, points: Tensor) -> Any:
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(points)
        return self.model(points)

    def forward(self, points: Tensor) -> Tensor:
        """Encode point clouds into a compact semidense embedding.

        Args:
            points: ``Tensor["B N 3", float32]`` semidense points (rig frame).

        Returns:
            ``Tensor["B out_dim", float32]`` semidense embeddings.
        """
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B,N,3), got {tuple(points.shape)}.")
        if points.shape[-1] != 3 and points.shape[1] == 3:
            points = points.transpose(1, 2)
        raw = _extract_tensor(self._forward_raw(points))
        if raw.ndim > 2:
            reduce_dims = tuple(range(2, raw.ndim))
            raw = raw.mean(dim=reduce_dims)
        raw = raw.to(dtype=self.proj.weight.dtype if isinstance(self.proj, nn.Linear) else raw.dtype)
        return self.proj(raw)


def _build_scene_field_v2(
    out: EvlBackboneOutput,
    *,
    occ_pr_is_logits: bool,
    scene_field_channels: list[str],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build a simplified scene field for VIN v2 (no hard thresholds).

    Channels (all float32, shape ``B x 1 x D x H x W``):
      - occ_pr: occupancy probability (sigmoid if logits).
      - cent_pr: centerness probability.
      - counts_norm: log1p-normalized observation counts in [0, 1].
      - occ_input: occupied evidence from input points (binary, no thresholding).
      - observed: soft observed mask (same as counts_norm).
      - unknown: soft unknown mask (1 - counts_norm).
      - free_input: EVL free-space if available, otherwise soft free derived from
        occ_input and counts_norm.
      - new_surface_prior: unknown * occ_pr.

    Returns:
        Tuple of (field_in, aux) where field_in contains only the requested
        scene_field_channels and aux provides all derived channels for
        internal use (e.g., coverage proxies).
    """

    def _require(name: str) -> Tensor:
        value = getattr(out, name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Missing backbone output '{name}'. Ensure EvlBackboneConfig.features_mode includes 'heads'.",
            )
        return value

    occ_pr = _require("occ_pr").to(dtype=torch.float32)
    if occ_pr_is_logits:
        occ_pr = torch.sigmoid(occ_pr)

    cent_pr = _require("cent_pr").to(dtype=torch.float32)
    occ_input = _require("occ_input").to(dtype=torch.float32)
    counts = _require("counts").to(dtype=torch.float32)

    max_counts = counts.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
    counts_norm = torch.log1p(counts).unsqueeze(1) / torch.log1p(max_counts).unsqueeze(
        1,
    )
    counts_norm = counts_norm.clamp(0.0, 1.0)
    unknown = (1.0 - counts_norm).clamp(0.0, 1.0)
    observed = counts_norm

    if isinstance(out.free_input, torch.Tensor):
        free_input = out.free_input.to(dtype=torch.float32)
    else:
        free_input = (1.0 - occ_input) * counts_norm

    new_surface_prior = unknown * occ_pr

    aux = {
        "occ_pr": occ_pr,
        "cent_pr": cent_pr,
        "counts_norm": counts_norm,
        "occ_input": occ_input,
        "observed": observed,
        "unknown": unknown,
        "free_input": free_input,
        "new_surface_prior": new_surface_prior,
    }
    field_parts = [aux[name] for name in scene_field_channels]
    return torch.cat(field_parts, dim=1), aux


def _ensure_candidate_batch(candidate_poses_world_cam: PoseTWT) -> PoseTWT:
    """Ensure candidate poses are batched as ``(B,N,12)``."""
    if candidate_poses_world_cam.ndim == 2:  # N x 12
        return PoseTW(candidate_poses_world_cam._data.unsqueeze(0))
    if candidate_poses_world_cam.ndim != 3:
        raise ValueError(
            "candidate_poses_world_cam must have shape (N,12) or (B,N,12).",
        )
    return candidate_poses_world_cam


def _ensure_pose_batch(pose: PoseTWT, *, batch_size: int, name: str) -> PoseTWT:
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


# TODO: Candidate‑relative positional keys: Build a pos_grid in the candidate frame so that queries and keys live in compatible coordinates (this is closer to “cross‑attention in the same frame”).
# TODO: for  stronger alignment between queries and keys try "Per‑axis normalization"
# i.e. pts_norm_axis = (pts_rig - center_rig) / (0.5 * span)


class PoseConditionedGlobalPool(nn.Module):
    """Pose-conditioned attention pooling over a coarse voxel grid.

    Conceptually, this module summarizes a dense voxel field into a compact
    per-candidate descriptor. It does so by:
      1) downsampling the voxel field into a fixed set of tokens,
      2) adding a learned positional embedding to those tokens, and
      3) using candidate pose embeddings as queries to attend over the tokens.

    Q/K/V usage:
      - **Queries (Q)**: projected candidate pose encodings (`q_proj(pose_enc)`).
      - **Keys (K)**: projected voxel field tokens plus positional embeddings
        (`kv_proj(field_tokens) + pos_proj(lff(pos_tokens))`).
      - **Values (V)**: projected voxel field tokens (`kv_proj(field_tokens)`).

    Positional embeddings are **only added to the keys**, not to the values, so
    the attention weights depend on both content and position while the values
    remain pure content summaries of the voxel field.
    """

    def __init__(
        self,
        *,
        field_dim: int,
        pose_dim: int,
        pool_size: int,
        num_heads: int,
        pos_grid_encoder: LearnableFourierFeaturesConfig,
    ) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0.")
        if field_dim % num_heads != 0:
            raise ValueError(
                f"field_dim ({field_dim}) must be divisible by num_heads ({num_heads}).",
            )

        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool3d(
            (self.pool_size, self.pool_size, self.pool_size),
        )
        self.kv_proj = nn.Linear(field_dim, field_dim)
        self.q_proj = nn.Linear(pose_dim, field_dim)
        self.pos_grid_encoder = pos_grid_encoder.setup_target()
        self.pos_proj = nn.Linear(self.pos_grid_encoder.out_dim, field_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=field_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, field: Tensor, pose_enc: Tensor, *, pos_grid: Tensor) -> Tensor:
        """Return pose-conditioned global tokens.

        Args:
            field: ``Tensor["B C D H W", float32]`` projected voxel field.
            pose_enc: ``Tensor["B N E", float32]`` pose embeddings.
            pos_grid: ``Tensor["B 3 D H W", float32]`` voxel position grid (normalized).

        Returns:
            ``Tensor["B N C", float32]`` pose-conditioned global features.
        """
        if field.ndim != 5:
            raise ValueError(
                f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.",
            )
        if pose_enc.ndim != 3:
            raise ValueError(
                f"Expected pose_enc shape (B,N,E), got {tuple(pose_enc.shape)}.",
            )
        if pos_grid.ndim != 5 or pos_grid.shape[1] != 3:
            raise ValueError(
                f"Expected pos_grid shape (B,3,D,H,W), got {tuple(pos_grid.shape)}.",
            )

        grid = min(
            self.pool_size,
            int(field.shape[-3]),
            int(field.shape[-2]),
            int(field.shape[-1]),
        )
        field_ds = functional.adaptive_avg_pool3d(field, output_size=(grid, grid, grid))
        tokens = field_ds.flatten(2).transpose(1, 2)  # B T C
        keys = self.kv_proj(tokens)

        pos_ds = functional.adaptive_avg_pool3d(
            pos_grid,
            output_size=(grid, grid, grid),
        )
        pos_tokens = pos_ds.flatten(2).transpose(1, 2)  # B T 3
        pos_enc = self.pos_grid_encoder(pos_tokens.to(dtype=keys.dtype))
        pos_emb = self.pos_proj(pos_enc)
        keys = keys + pos_emb
        queries = self.q_proj(pose_enc.to(dtype=keys.dtype))
        attn_out, _ = self.attn(queries, keys, keys, need_weights=False)
        return attn_out


# TODO: add learned threholds for fields channels - i.e. cent_pr contains many low-confidence values that are irrelevant
class VinModelV2Config(BaseConfig["VinModelV2"]):
    """Configuration for :class:`VinModelV2` (minimal, configurable)."""

    target: type[VinModelV2] = Field(default_factory=lambda: VinModelV2, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Optional frozen EVL backbone configuration."""

    pose_encoder: PoseEncoderConfig = Field(default_factory=R6dLffPoseEncoderConfig)
    """Pose encoder configuration (discriminated union).

    Note: shell-based encoders use only the forward direction and therefore do
    not encode roll about the forward axis; this is acceptable when roll jitter
    is small.
    """

    pos_grid_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(
            input_dim=3,
            fourier_dim=32,
            hidden_dim=32,
            output_dim=16,
        ),
    )
    """LFF encoder for XYZ voxel position keys (input_dim=3)."""

    head_hidden_dim: int = optimizable_field(
        default=128,
        optimizable=Optimizable.discrete(
            low=64,
            high=512,
            step=64,
            description="Hidden dimension for the scorer MLP.",
        ),
        gt=0,
    )
    """Hidden dimension for the scorer MLP."""

    head_num_layers: int = optimizable_field(
        default=1,
        optimizable=Optimizable.discrete(
            low=1,
            high=3,
            step=1,
            description="Number of MLP layers before the CORAL layer.",
        ),
        ge=1,
    )
    """Number of MLP layers before the CORAL layer."""

    head_dropout: float = optimizable_field(
        default=0.0,
        optimizable=Optimizable.continuous(
            low=0.0,
            high=0.4,
            description="Dropout probability in the MLP.",
        ),
        ge=0.0,
        lt=1.0,
    )
    """Dropout probability in the MLP."""

    head_activation: Literal["gelu", "relu"] = optimizable_field(
        default="gelu",
        optimizable=Optimizable.categorical(
            choices=("gelu", "relu"),
            description="Activation function for the scorer MLP.",
        ),
    )
    """Activation function ('gelu' or 'relu')."""

    num_classes: int = Field(default=15, ge=2)
    """Number of ordinal bins (VIN-NBV uses 15)."""

    coral_preinit_bias: bool = True
    """ If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice."""

    field_dim: int = optimizable_field(
        default=32,
        optimizable=Optimizable.discrete(
            low=16,
            high=128,
            step=16,
            description="Channel dimension of the projected voxel field.",
        ),
        gt=0,
    )
    """Channel dimension of the projected voxel field."""

    field_gn_groups: int = optimizable_field(
        default=4,
        optimizable=Optimizable.discrete(
            low=1,
            high=8,
            step=1,
            description="GroupNorm groups for the projected voxel field.",
        ),
        gt=0,
    )
    """Requested GroupNorm groups (clamped to a divisor of ``field_dim``)."""

    point_encoder: PointNeXtSEncoderConfig | None = None
    """Optional PointNeXt-S encoder for semidense point cloud features."""

    occ_pr_is_logits: bool = False
    """Whether EVL ``occ_pr`` is logits (apply sigmoid) rather than probabilities."""

    apply_cw90_correction: bool = True
    """Undo ``rotate_yaw_cw90`` on candidate/reference poses + cameras."""

    # TODO: not implemented yet
    tf_pos_grid_in_candidate_frame: bool = False
    """If True, transform the voxel positions into each candidate frame for positional keys rather than the reference rig frame."""

    global_pool_grid_size: int = optimizable_field(
        default=6,
        optimizable=Optimizable.discrete(
            low=4,
            high=8,
            step=1,
            description="Target grid size for pose-conditioned global pooling.",
        ),
        gt=0,
    )
    """Target grid size for pose-conditioned global pooling."""

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
    ] = optimizable_field(
        default_factory=lambda: [
            "occ_pr",
            "new_surface_prior",
        ],
        optimizable=Optimizable.categorical(
            choices=(
                ["occ_pr"],
                ["occ_pr", "counts_norm"],
                ["occ_pr", "unknown", "new_surface_prior"],
                ["occ_pr", "occ_input", "free_input", "counts_norm"],
            ),
            description="Scene-field channel selection for the voxel field.",
        ),
        min_length=1,
    )
    """Ordered list of scene-field channels to include in the voxel field."""

    @field_validator("pos_grid_encoder_lff")
    @classmethod
    def _validate_pos_grid_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        if value.input_dim != 3:
            raise ValueError(
                "pos_grid_encoder_lff.input_dim must be 3 for XYZ coordinates.",
            )
        return value

    @field_validator("scene_field_channels")
    @classmethod
    def _validate_scene_field_channels(cls, value: list[str]) -> list[str]:
        """Validate requested scene-field channels.

        We only allow channels that can be constructed from EVL head outputs in
        this module, ensuring the scene field definition remains explicit and
        interpretable (e.g., ``new_surface_prior = unknown * occ_pr``).
        """
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


class VinModelV2(nn.Module):
    """Simplified VIN head for RRI prediction with configurable pose encoding."""

    def __init__(self, config: VinModelV2Config) -> None:
        super().__init__()
        self.config = config
        # Lazily initialize the backbone on first forward if needed.
        self.backbone = None
        self.pose_encoder: PoseEncoder = self.config.pose_encoder.setup_target()
        self.point_encoder: PointNeXtSEncoder | None = (
            self.config.point_encoder.setup_target() if self.config.point_encoder is not None else None
        )

        field_dim = int(self.config.field_dim)
        gn_groups = _largest_divisor_leq(field_dim, int(self.config.field_gn_groups))
        self.field_proj = nn.Sequential(
            nn.Conv3d(
                len(self.config.scene_field_channels),
                field_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )

        pose_dim = int(self.pose_encoder.out_dim)
        num_heads = _largest_divisor_leq(field_dim, 4)
        self.global_pooler = PoseConditionedGlobalPool(
            field_dim=field_dim,
            pose_dim=pose_dim,
            pool_size=int(self.config.global_pool_grid_size),
            num_heads=num_heads,
            pos_grid_encoder=self.config.pos_grid_encoder_lff,
        )

        point_dim = int(self.point_encoder.out_dim) if self.point_encoder is not None else 0
        head_in_dim = pose_dim + field_dim + point_dim
        act: nn.Module
        match self.config.head_activation:
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()

        hidden_dim = int(self.config.head_hidden_dim)
        layers: list[nn.Module] = [nn.Linear(head_in_dim, hidden_dim), act]
        if self.config.head_dropout > 0:
            layers.append(nn.Dropout(p=float(self.config.head_dropout)))
        for _ in range(int(self.config.head_num_layers) - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if self.config.head_dropout > 0:
                layers.append(nn.Dropout(p=float(self.config.head_dropout)))

        self.head_mlp = nn.Sequential(*layers)
        self.head_coral = CoralLayer(
            in_dim=hidden_dim,
            num_classes=self.config.num_classes,
            preinit_bias=self.config.coral_preinit_bias,
        )
        device = self.backbone.device if self.backbone is not None else torch.device("cpu")
        self.to(device)

    @property
    def pose_encoder_lff(self) -> LearnableFourierFeatures | None:
        """Return the LFF encoder when the pose encoder uses LFF (else ``None``)."""
        return getattr(self.pose_encoder, "pose_encoder_lff", None)

    @staticmethod
    def _pos_grid_from_pts_world(
        pts_world: Tensor,
        *,
        t_world_voxel: PoseTW,
        pose_world_rig_ref: PoseTW,
        voxel_extent: Tensor,
        grid_shape: tuple[int, int, int],
    ) -> Tensor:
        """Convert voxel center points to a normalized position grid in the reference rig frame.

        Only the translational positions are encoded. ``pos_grid`` is the 3D
        coordinates of voxel centers in the reference rig frame (normalized);
        LFF is applied inside the global pooler for positional keys.


        Args:
            pts_world: ``Tensor["B (D·H·W) 3"]`` or ``Tensor["B D H W 3"]``.
            t_world_voxel: ``PoseTW["B 12"]`` world←voxel pose.
            pose_world_rig_ref: ``PoseTW["B 12"]`` world←rig pose defining the reference frame.
            voxel_extent: ``Tensor["B 6"]`` voxel extent in voxel frame (used for scale).
            grid_shape: Tuple ``(D,H,W)`` matching the field resolution.

        Returns:
            ``Tensor["B 3 D H W", float32]`` normalized voxel coordinates in [-1, 1].
        """
        if pts_world.ndim == 3:
            batch_size, num_pts, _ = pts_world.shape
            expected = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
            if num_pts != expected:
                raise ValueError(
                    f"pts_world has {num_pts} points, expected {expected} from grid_shape {grid_shape}.",
                )
            pts_grid = pts_world.view(
                batch_size,
                grid_shape[0],
                grid_shape[1],
                grid_shape[2],
                3,
            )
        elif pts_world.ndim == 5:
            pts_grid = pts_world
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
        scale = 0.5 * span.max(dim=-1, keepdim=True).values.clamp_min(1e-6)

        center_world = t_world_voxel * center_vox
        center_rig = t_rig_world * center_world
        pts_norm = (pts_rig - center_rig[:, None, :]) / scale[:, None, :]

        pts_norm = pts_norm.view(
            pts_grid.shape[0],
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            3,
        )
        return pts_norm.permute(0, 4, 1, 2, 3).contiguous()

    def _semidense_points_world(self, efm: dict[str, Any]) -> Tensor | None:
        """Collapse semidense points across time (subsampled)."""
        points = efm.get(ARIA_POINTS_WORLD)
        if not isinstance(points, torch.Tensor):
            return None
        dist_std = efm.get(ARIA_POINTS_DIST_STD)
        if not isinstance(dist_std, torch.Tensor):
            dist_std = torch.zeros_like(points[..., 0])
        inv_dist_std = efm.get(ARIA_POINTS_INV_DIST_STD)
        if not isinstance(inv_dist_std, torch.Tensor):
            inv_dist_std = torch.zeros_like(points[..., 0])
        time_ns = efm.get(ARIA_POINTS_TIME_NS)
        if not isinstance(time_ns, torch.Tensor):
            time_ns = torch.zeros((points.shape[0],), device=points.device, dtype=torch.int64)
        vol_min = efm.get(ARIA_POINTS_VOL_MIN) or efm.get("points/vol_min")
        if not isinstance(vol_min, torch.Tensor):
            vol_min = torch.zeros((3,), device=points.device, dtype=points.dtype)
        vol_max = efm.get(ARIA_POINTS_VOL_MAX) or efm.get("points/vol_max")
        if not isinstance(vol_max, torch.Tensor):
            vol_max = torch.zeros((3,), device=points.device, dtype=points.dtype)
        lengths = efm.get("points/lengths") or efm.get("msdpd#points_world_lengths")
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.full(
                (points.shape[0],),
                points.shape[1],
                dtype=torch.int64,
                device=points.device,
            )

        points_view = EfmPointsView(
            points_world=points,
            dist_std=dist_std,
            inv_dist_std=inv_dist_std,
            time_ns=time_ns,
            volume_min=vol_min,
            volume_max=vol_max,
            lengths=lengths,
        )
        max_points = self.config.point_encoder.max_points if self.config.point_encoder is not None else None
        return points_view.collapse_points(max_points=max_points)

    def _forward_impl(
        self,
        efm: dict[str, Any],
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        return_debug: bool,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinV2ForwardDiagnostics | None]:
        """Run the VIN v2 forward pass."""
        if backbone_out is None:
            if self.backbone is None:  # type: ignore
                self.backbone = self.config.backbone.setup_target() if self.config.backbone is not None else None
            backbone_out = self.backbone.forward(efm)  # type: ignore
        device = backbone_out.voxel_extent.device
        if next(self.parameters()).device != device:
            self.to(device)
        p3d_cameras = p3d_cameras.to(device)

        pose_world_cam = _ensure_candidate_batch(candidate_poses_world_cam).to(
            device=device,
        )  # type: ignore[arg-type]
        batch_size, num_candidates = (
            int(pose_world_cam.shape[0]),
            int(pose_world_cam.shape[1]),
        )

        pose_world_rig_ref = _ensure_pose_batch(
            reference_pose_world_rig.to(device=device),  # type: ignore[arg-type]
            batch_size=batch_size,
            name="reference_pose_world_rig",
        )

        if self.config.apply_cw90_correction:
            pose_world_cam = rotate_yaw_cw90(pose_world_cam, undo=True)
            pose_world_rig_ref = rotate_yaw_cw90(pose_world_rig_ref, undo=True)
        _ = p3d_cameras

        # ------------------------------------------------------------------ relative pose (candidate in reference rig frame)
        pose_rig_cam = pose_world_rig_ref.inverse()[:, None] @ pose_world_cam  # rig_ref <- cam

        # ------------------------------------------------------------------ pose encoding (configurable)
        pose_out = self.pose_encoder.encode(pose_rig_cam)
        pose_enc = pose_out.pose_enc
        pose_vec = pose_out.pose_vec

        candidate_center_rig_m = pose_out.center_m

        # ------------------------------------------------------------------ voxel pose (for positional keys only)
        t_world_voxel = backbone_out.t_world_voxel
        t_world_voxel = _ensure_pose_batch(
            t_world_voxel,
            batch_size=batch_size,
            name="voxel/T_world_voxel",
        )

        # ------------------------------------------------------------------ build voxel-aligned scene field
        field_in, field_aux = _build_scene_field_v2(
            backbone_out,
            occ_pr_is_logits=self.config.occ_pr_is_logits,
            scene_field_channels=self.config.scene_field_channels,
        )
        field_in = field_in.to(device=device)
        field = self.field_proj(field_in)

        # ------------------------------------------------------------------ candidate validity + coverage proxy
        candidate_centers_world = pose_world_cam.t.to(
            dtype=field.dtype,
        )  # B N 3 (world frame)
        counts_norm = field_aux["counts_norm"].to(device=device)
        center_tokens, center_valid = _sample_voxel_field(
            counts_norm,
            points_world=candidate_centers_world.unsqueeze(2),  # B N 1 3
            t_world_voxel=t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        center_valid = center_valid.squeeze(-1)
        counts_norm_center = center_tokens[..., 0, 0]

        # ------------------------------------------------------------------ global pooling (pose-conditioned token + positional keys)
        pts_world = backbone_out.pts_world
        if not isinstance(pts_world, torch.Tensor):
            raise KeyError(
                "Missing backbone output 'voxel/pts_world' required for positional encoding.",
            )
        # TODO: we only need to encode the translational component of the pos_grid (no need for the rot components)
        pos_grid = self._pos_grid_from_pts_world(
            pts_world.to(device=device, dtype=field.dtype),
            t_world_voxel=t_world_voxel,
            pose_world_rig_ref=pose_world_rig_ref,
            voxel_extent=backbone_out.voxel_extent,
            grid_shape=(field.shape[-3], field.shape[-2], field.shape[-1]),
        )
        global_feat = self.global_pooler(field, pose_enc, pos_grid=pos_grid).to(
            dtype=field.dtype,
        )

        # ------------------------------------------------------------------ candidate validity + coverage weighting
        pose_finite = torch.isfinite(pose_vec).all(dim=-1)
        candidate_valid = pose_finite & center_valid
        voxel_valid_frac = counts_norm_center * center_valid.to(
            dtype=counts_norm_center.dtype,
        )
        voxel_valid_frac = (voxel_valid_frac * pose_finite.to(dtype=voxel_valid_frac.dtype)).clamp(
            0.0,
            1.0,
        )

        semidense_feat = None
        if self.point_encoder is not None:
            pts_world = self._semidense_points_world(efm)
            if pts_world is None or pts_world.numel() == 0:
                semidense_feat = torch.zeros(
                    (batch_size, self.point_encoder.out_dim),
                    device=device,
                    dtype=field.dtype,
                )
            else:
                pts_world = pts_world.to(device=device, dtype=torch.float32)
                t_rig_world = pose_world_rig_ref.inverse()
                pts_rig = t_rig_world * pts_world  # B x K x 3
                if pts_rig.ndim == 2:
                    pts_rig = pts_rig.unsqueeze(0).expand(batch_size, -1, -1)
                semidense_feat = self.point_encoder(pts_rig.to(device=device))
            semidense_feat = semidense_feat.to(device=device, dtype=field.dtype)

        parts: list[Tensor] = [
            pose_enc.to(device=device, dtype=field.dtype),
            global_feat,
        ]
        if semidense_feat is not None:
            parts.append(semidense_feat[:, None, :].expand(batch_size, num_candidates, -1))

        feats = torch.cat(parts, dim=-1)
        flat_feats = feats.reshape(batch_size * num_candidates, -1)
        logits = self.head_coral(self.head_mlp(flat_feats)).reshape(
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
            voxel_valid_frac=voxel_valid_frac.to(dtype=field.dtype),
            semidense_valid_frac=None,
        )

        if not return_debug:
            return pred, None

        debug = VinV2ForwardDiagnostics(
            backbone_out=backbone_out,
            candidate_center_rig_m=candidate_center_rig_m,
            pose_enc=pose_enc,
            pose_vec=pose_vec,
            field_in=field_in,
            field=field,
            global_feat=global_feat,
            candidate_valid=candidate_valid,
            voxel_valid_frac=voxel_valid_frac.to(dtype=field.dtype),
            feats=feats,
        )
        return pred, debug

    def forward(
        self,
        efm: dict[str, Any],
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> VinPrediction:
        """Score candidate poses for one snippet (no diagnostics)."""
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
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinV2ForwardDiagnostics]:
        """Run VIN v2 forward pass and return intermediate tensors."""
        pred, debug = self._forward_impl(
            efm,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            p3d_cameras=p3d_cameras,
            return_debug=True,
            backbone_out=backbone_out,
        )
        if debug is None:
            raise RuntimeError(
                "Expected VinV2ForwardDiagnostics when return_debug=True.",
            )
        return pred, debug

    def init_bin_values(self, values: Tensor, *, overwrite: bool = False) -> None:
        """Initialize learnable bin representatives for CORAL expectation.

        Args:
            values: ``Tensor["K"]`` target bin representatives (e.g., bin means).
            overwrite: If True, overwrite existing bin values.
        """
        self.head_coral.init_bin_values(values, overwrite=overwrite)

    def summarize_vin(
        self,
        batch: VinOracleBatch,
        *,
        include_torchsummary: bool = True,
        torchsummary_depth: int = 3,
    ) -> str:
        """Summarize VIN v2 inputs/outputs for a single oracle-labeled batch."""
        from efm3d.aria.aria_constants import (
            ARIA_CALIB,
            ARIA_IMG,
            ARIA_POSE_T_WORLD_RIG,
        )

        from oracle_rri.utils import Console
        from oracle_rri.utils.rich_summary import rich_summary, summarize

        def _capture_tree(tree) -> str:
            console = Console()
            with console.capture() as capture:
                console.print(
                    tree,
                    soft_wrap=False,
                    highlight=True,
                    markup=True,
                    emoji=False,
                )
            return capture.get().rstrip()

        if batch.efm_snippet_view is None and batch.backbone_out is None:
            raise RuntimeError(
                "VIN v2 summary requires efm inputs or cached backbone outputs.",
            )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            efm = batch.efm_snippet_view.efm if batch.efm_snippet_view is not None else {}
            pred, debug = self.forward_with_debug(
                efm,
                candidate_poses_world_cam=batch.candidate_poses_world_cam,
                reference_pose_world_rig=batch.reference_pose_world_rig,
                p3d_cameras=batch.p3d_cameras,
                backbone_out=batch.backbone_out,
            )
        if was_training:
            self.train()

        efm = batch.efm_snippet_view.efm if batch.efm_snippet_view is not None else {}
        backbone_out = debug.backbone_out
        if backbone_out is None:
            raise RuntimeError(
                "VIN v2 summary expected backbone outputs to be available.",
            )
        if batch.efm_snippet_view is None:
            efm_summary = {"note": "cached batch (raw EFM inputs unavailable)"}
        else:
            efm_summary = {
                **{key: summarize(efm.get(key)) for key in ARIA_IMG},
                **{key: summarize(efm.get(key)) for key in ARIA_CALIB},
                ARIA_POSE_T_WORLD_RIG: summarize(efm.get(ARIA_POSE_T_WORLD_RIG)),
            }
        summary_dict = {
            "meta": {
                "scene_id": batch.scene_id,
                "snippet_id": batch.snippet_id,
                "device": str(debug.candidate_center_rig_m.device),
                "candidates": summarize(batch.candidate_poses_world_cam),
            },
            "efm": efm_summary,
            "backbone": {
                "occ_pr": summarize(backbone_out.occ_pr),
                "occ_input": summarize(backbone_out.occ_input),
                "counts": summarize(backbone_out.counts),
                "cent_pr": summarize(backbone_out.cent_pr),
                "voxel/pts_world": summarize(backbone_out.pts_world),
                "T_world_voxel": summarize(backbone_out.t_world_voxel),
                "voxel_extent": summarize(backbone_out.voxel_extent),
            },
            "pose": {
                "candidate_center_rig_m": summarize(
                    debug.candidate_center_rig_m,
                    include_stats=True,
                ),
                "pose_vec": summarize(debug.pose_vec, include_stats=True),
                "pose_enc": summarize(debug.pose_enc),
            },
            "features": {
                "field_in": summarize(debug.field_in),
                "field": summarize(debug.field),
                "global_feat": summarize(debug.global_feat),
                "concat_feats": summarize(debug.feats),
            },
            "validity": {
                "candidate_valid": summarize(debug.candidate_valid),
                "voxel_valid_frac": summarize(pred.voxel_valid_frac, include_stats=True),
            },
            "outputs": {
                "logits": summarize(pred.logits),
                "prob": summarize(pred.prob),
                "expected": summarize(pred.expected, include_stats=True),
                "expected_normalized": summarize(
                    pred.expected_normalized,
                    include_stats=True,
                ),
            },
        }
        for key in ["points/p3s_world", "points/dist_std", "pose/gravity_in_world"]:
            if key in efm:
                summary_dict.setdefault("efm", {})[key] = summarize(efm.get(key))

        tree = rich_summary(
            tree_dict=summary_dict,
            root_label="VIN v2 summary (oracle batch)",
            with_shape=True,
            is_print=False,
        )
        lines: list[str] = [_capture_tree(tree), ""]

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        lines.append(
            f"Trainable VIN params: {trainable_params:,} (vin total params: {total_params:,}; EVL frozen not counted)",
        )
        lines.append("")

        if include_torchsummary:
            from torchsummary import summary as torch_summary

            pose_vec = debug.pose_vec.reshape(
                debug.pose_vec.shape[0] * debug.pose_vec.shape[1],
                -1,
            )
            feats_2d = debug.feats.reshape(
                debug.feats.shape[0] * debug.feats.shape[1],
                -1,
            )

            pose_encoder_lff = self.pose_encoder_lff
            if pose_encoder_lff is not None:
                lines.append("torchsummary: pose_encoder_lff (trainable)")
                lines.append(
                    str(
                        torch_summary(
                            pose_encoder_lff,
                            input_data=pose_vec,
                            verbose=0,
                            depth=torchsummary_depth,
                            device=debug.candidate_center_rig_m.device,
                        ),
                    ),
                )
                lines.append("")
            else:
                lines.append("torchsummary: pose_encoder (non-LFF) skipped")
                lines.append("")

            lines.append("torchsummary: field_proj (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.field_proj,
                        input_data=debug.field_in,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    ),
                ),
            )
            lines.append("")

            lines.append("torchsummary: scorer MLP (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.head_mlp,
                        input_data=feats_2d,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    ),
                ),
            )
            lines.append("")

            lines.append("torchsummary: CORAL head (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.head_coral,
                        input_data=self.head_mlp(feats_2d),
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    ),
                ),
            )

        return "\n".join(lines)

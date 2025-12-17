"""VIN model on top of a frozen EVL backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from efm3d.aria.aria_constants import ARIA_POSE_T_WORLD_RIG
from efm3d.aria.pose import PoseTW
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field
from torch import Tensor, nn

from ..utils import BaseConfig
from .backbone_evl import EvlBackboneConfig
from .coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob
from .pose_encoding import LearnableFourierFeaturesConfig
from .spherical_encoding import ShellShPoseEncoderConfig
from .types import EvlBackboneOutput, VinPrediction


def _first_key(key: str | Sequence[str]) -> str:
    if isinstance(key, (list, tuple)):
        return str(key[0])
    return str(key)


def _largest_divisor_leq(n: int, max_divisor: int) -> int:
    g = min(int(max_divisor), int(n))
    while g > 1 and (n % g) != 0:
        g -= 1
    return max(1, g)


def _build_frustum_points_cam(
    *,
    grid_size: int,
    depths_m: list[float],
    fov_deg: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if grid_size <= 0:
        raise ValueError("grid_size must be > 0.")
    if not depths_m:
        raise ValueError("depths_m must not be empty.")
    if fov_deg <= 0:
        raise ValueError("fov_deg must be > 0.")

    # NOTE: For v0.1 we use a simple pinhole-style directional grid defined by a single
    # symmetric FOV prior. This intentionally ignores per-snippet intrinsics; replace with
    # intrinsics-aware unprojection if needed later.
    s = math.tan(math.radians(float(fov_deg)) / 2.0)
    xy = torch.linspace(-s, s, steps=int(grid_size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")

    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)  # G 3
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    depths = torch.tensor(depths_m, device=device, dtype=dtype).reshape(-1, 1)  # D 1
    points = (dirs.unsqueeze(0) * depths.unsqueeze(1)).reshape(-1, 3)  # (D*G) 3
    return points


def _make_token_positions(size: int, *, device: torch.device | None = None) -> Tensor:
    coords = torch.linspace(-1.0, 1.0, steps=int(size), device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    pos = torch.stack([xx, yy, zz], dim=-1)  # D H W 3
    return pos.reshape(-1, 3)


def _build_scene_field(
    out: EvlBackboneOutput,
    *,
    use_channels: list[str],
    occ_input_threshold: float,
    counts_norm_mode: Literal["log1p", "linear"],
    occ_pr_is_logits: bool,
) -> Tensor:
    """Build a low-dimensional scene field from EVL head outputs.

    Args:
        out: Backbone output bundle (must include head/evidence tensors).
        use_channels: Ordered list of channels to include.
        occ_input_threshold: Threshold used when deriving fallback free-space evidence.
        counts_norm_mode: Normalization mode for counts.
        occ_pr_is_logits: Whether `occ_pr` are logits (apply sigmoid) rather than probabilities.

    Returns:
        Tensor["B C D H W", float32] scene field.
    """

    if not use_channels:
        raise ValueError("use_channels must not be empty.")

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
        if occ_pr_is_logits:
            occ_pr = torch.sigmoid(occ_pr)
        parts["occ_pr"] = occ_pr

    if "occ_input" in use_channels or "free_input" in use_channels:
        parts["occ_input"] = _require("occ_input").to(dtype=torch.float32)

    if "free_input" in use_channels:
        if isinstance(out.free_input, torch.Tensor):
            parts["free_input"] = out.free_input.to(dtype=torch.float32)
        else:
            # Fallback: derive a weak free-space proxy from (counts, occ_input).
            counts = _require("counts")
            observed = (counts > 0).to(dtype=torch.float32).unsqueeze(1)
            occ_evidence = (parts["occ_input"] > float(occ_input_threshold)).to(dtype=torch.float32)
            parts["free_input"] = observed * (1.0 - occ_evidence)

    if (
        "counts_norm" in use_channels
        or "observed" in use_channels
        or "unknown" in use_channels
        or "new_surface_prior" in use_channels
    ):
        counts = _require("counts").to(dtype=torch.float32)
        observed = (counts > 0).to(dtype=torch.float32)
        parts["observed"] = observed.unsqueeze(1)
        parts["unknown"] = (1.0 - observed).unsqueeze(1)

        max_counts = counts.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        match str(counts_norm_mode):
            case "log1p":
                parts["counts_norm"] = torch.log1p(counts).unsqueeze(1) / torch.log1p(max_counts).unsqueeze(1)
            case "linear":
                parts["counts_norm"] = (counts / max_counts).unsqueeze(1)
            case other:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported counts_norm_mode: {other}")

    if "new_surface_prior" in use_channels:
        parts["new_surface_prior"] = parts["unknown"] * parts["occ_pr"]

    field_parts: list[Tensor] = []
    for name in use_channels:
        if name not in parts:
            raise KeyError(f"Unknown/unsupported scene-field channel: {name!r}")
        field_parts.append(parts[name])
    return torch.cat(field_parts, dim=1)


def _sample_voxel_field(
    field: Tensor,
    *,
    points_world: Tensor,
    t_world_voxel: PoseTW,
    voxel_extent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sample a voxel-aligned field at world points.

    Returns:
        - tokens: ``Tensor["B N K C", float32]``
        - valid: ``Tensor["B N K", bool]``
    """

    if field.ndim != 5:
        raise ValueError(f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.")
    if points_world.ndim != 4:
        raise ValueError(f"Expected points_world shape (B,N,K,3), got {tuple(points_world.shape)}.")
    if int(points_world.shape[-1]) != 3:
        raise ValueError(f"Expected points_world[..., 3], got {tuple(points_world.shape)}.")

    batch_size, field_channels, grid_d, grid_h, grid_w = field.shape
    _, num_candidates, num_points, _ = points_world.shape

    t_world_voxel_b = t_world_voxel
    if t_world_voxel_b.ndim == 1:
        t_world_voxel_b = PoseTW(t_world_voxel_b._data.unsqueeze(0))
    if int(t_world_voxel_b.shape[0]) != int(batch_size):
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

    # NOTE: EVL's voxel field is defined in the *voxel frame* (metres), but our candidates/frustum points are in WORLD.
    # EVL provides `voxel/T_world_voxel` (world←voxel). We invert it to get voxel←world and map points into voxel coords.
    # FIXME: If you ever swap EVL conventions or change the voxel-grid anchoring, this is the one transform you must
    # re-verify (sanity check: voxelized points should be stable under small candidate translations).
    t_voxel_world = t_world_voxel_b.inverse()  # voxel<-world
    voxel_points_m = t_voxel_world * world_points_flat  # B (N*K) 3 in voxel frame (metres)

    pts_vox_id, valid_extent = pc_to_vox(
        voxel_points_m.to(dtype=torch.float32),
        vW=int(grid_w),
        vH=int(grid_h),
        vD=int(grid_d),
        voxel_extent=vox_extent,
    )
    # sample_voxels does not support NaNs; replace invalid coords with 0 and rely on validity masks below.
    pts_vox_id = torch.nan_to_num(pts_vox_id, nan=0.0, posinf=0.0, neginf=0.0)

    samp, valid_grid = sample_voxels(field, pts_vox_id, differentiable=False)  # B C (N*K), B (N*K)
    valid = (valid_extent & valid_grid).reshape(batch_size, num_candidates, num_points)
    tokens = samp.transpose(1, 2).reshape(batch_size, num_candidates, num_points, field_channels)
    return tokens, valid


class VinScorerHead(nn.Module):
    """Candidate scoring head producing CORAL logits."""

    def __init__(
        self,
        *,
        in_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")

        act: nn.Module
        match activation.lower():
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()
            case other:
                raise ValueError(f"Unsupported activation: {other}")

        layers: list[nn.Module] = []
        if in_dim is None:
            layers.append(nn.LazyLinear(hidden_dim))
        else:
            layers.append(nn.Linear(int(in_dim), hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))

        self.mlp = nn.Sequential(*layers)
        self.coral = CoralLayer(in_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.coral(self.mlp(x))


class VinScorerHeadConfig(BaseConfig[VinScorerHead]):
    """Configuration for :class:`VinScorerHead`."""

    target: type[VinScorerHead] = Field(default_factory=lambda: VinScorerHead, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    hidden_dim: int = 128
    """Hidden dimension for MLP layers."""

    num_layers: int = 1
    """Number of MLP layers before the CORAL layer."""

    dropout: float = 0.0
    """Dropout probability in the MLP."""

    num_classes: int = 15
    """Number of ordinal bins (VIN-NBV uses 15)."""

    activation: str = "gelu"
    """Activation function ('gelu' or 'relu')."""

    def setup_target(self, *, in_dim: int | None = None) -> VinScorerHead:  # type: ignore[override]
        return self.target(
            in_dim=in_dim,
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
            num_classes=int(self.num_classes),
            activation=str(self.activation),
        )


def _vin_target() -> type["VinModel"]:
    return VinModel


class VinModelConfig(BaseConfig["VinModel"]):
    """Configuration for :class:`VinModel`."""

    target: type["VinModel"] = Field(default_factory=_vin_target, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoding_mode: Literal["shell_sh", "lff6d"] = "shell_sh"
    """Pose encoding mode.

    - ``shell_sh``: shell descriptor + spherical harmonics ($u,f$) and 1D Fourier features (radius).
    - ``lff6d``: learnable Fourier features baseline on a 6D descriptor ``[t, f]``.
    """

    pose_encoder_sh: ShellShPoseEncoderConfig = Field(default_factory=ShellShPoseEncoderConfig)
    """Spherical harmonics pose encoding configuration (shell descriptor)."""

    pose_encoder: LearnableFourierFeaturesConfig = Field(default_factory=LearnableFourierFeaturesConfig)
    """Learnable Fourier features configuration (baseline for 6D pose descriptor)."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: list[str] = Field(
        default_factory=lambda: ["occ_pr", "occ_input", "free_input", "counts_norm", "unknown", "new_surface_prior"]
    )
    """Ordered channels used to build the low-dimensional scene field."""

    occ_input_threshold: float = 0.5
    """Threshold used when deriving fallback free-space evidence from `occ_input`."""

    counts_norm_mode: Literal["log1p", "linear"] = "log1p"
    """How to normalize voxel `counts` into [0, 1]."""

    occ_pr_is_logits: bool = False
    """Whether `occ_pr` is logits (apply sigmoid) rather than a probability volume."""

    field_dim: int = 16
    """Channel dimension d0 of the compressed scene field."""

    field_gn_groups: int = 4
    """Requested GroupNorm groups for the field projection (clamped to a divisor of `field_dim`)."""

    global_token_grid_size: int = 4
    """Coarse global token grid size (default: 4 → 64 tokens)."""

    global_num_queries: int = 1
    """Number of learnable query tokens for global attention pooling."""

    global_num_heads: int = 2
    """Multi-head attention heads for global pooling."""

    global_attn_dropout: float = 0.0
    """Attention dropout probability for global pooling."""

    global_use_positional_encoding: bool = True
    """Add a lightweight 3D positional embedding to global tokens when True."""

    frustum_grid_size: int = 4
    """Grid size on the image plane for candidate frustum sampling (grid_size² directions)."""

    frustum_depths_m: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])
    """Depth values (metres) along each frustum direction."""

    frustum_fov_deg: float = 90.0
    """Approximate symmetric FOV used for the candidate frustum sampling grid."""

    candidate_num_heads: int = 2
    """Multi-head attention heads for candidate cross-attention pooling."""

    candidate_attn_dropout: float = 0.0
    """Attention dropout probability for candidate pooling."""

    use_global_pool: bool = True
    """Whether to concatenate the global pooled embedding to per-candidate features."""


class VinModel(nn.Module):
    """View Introspection Network (VIN) predicting RRI from EVL voxel features + candidate pose."""

    def __init__(self, config: VinModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = self.config.backbone.setup_target()
        self.pose_encoder_lff = self.config.pose_encoder.setup_target()
        self.pose_encoder_sh = self.config.pose_encoder_sh.setup_target()
        self._freeze_inactive_pose_encoder()

        field_dim = int(self.config.field_dim)
        if field_dim <= 0:
            raise ValueError("field_dim must be > 0.")
        gn_groups = _largest_divisor_leq(field_dim, int(self.config.field_gn_groups))

        self.field_proj = nn.Sequential(
            nn.LazyConv3d(field_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=int(gn_groups), num_channels=int(field_dim)),
            nn.GELU(),
        )

        if field_dim % int(self.config.candidate_num_heads) != 0:
            raise ValueError("field_dim must be divisible by candidate_num_heads.")
        self.cand_pose_to_query = nn.LazyLinear(field_dim)
        self.cand_attn = nn.MultiheadAttention(
            embed_dim=int(field_dim),
            num_heads=int(self.config.candidate_num_heads),
            dropout=float(self.config.candidate_attn_dropout),
            batch_first=True,
        )

        self.global_token_grid_size = int(self.config.global_token_grid_size)
        self.global_use_positional_encoding = bool(self.config.global_use_positional_encoding)
        self.global_num_queries = int(self.config.global_num_queries)

        if self.config.use_global_pool:
            if field_dim % int(self.config.global_num_heads) != 0:
                raise ValueError("field_dim must be divisible by global_num_heads.")
            self.global_attn = nn.MultiheadAttention(
                embed_dim=int(field_dim),
                num_heads=int(self.config.global_num_heads),
                dropout=float(self.config.global_attn_dropout),
                batch_first=True,
            )
            self.global_query = nn.Parameter(torch.randn((self.global_num_queries, field_dim)) * 0.02)
            if self.global_use_positional_encoding:
                pos = _make_token_positions(self.global_token_grid_size)
                self.register_buffer("_global_token_pos", pos, persistent=False)
                self.global_pos_mlp = nn.Linear(3, int(field_dim), bias=False)
            else:
                self.register_buffer("_global_token_pos", torch.empty((0, 3)), persistent=False)
                self.global_pos_mlp = None
        else:
            self.global_attn = None
            self.global_query = None
            self.register_buffer("_global_token_pos", torch.empty((0, 3)), persistent=False)
            self.global_pos_mlp = None

        # NOTE: Register the canonical frustum sample points as a buffer so it moves with `.to(...)`.
        # The points are deterministic (derived from config), hence `persistent=False`.
        self._frustum_points_cam: Tensor
        frustum_points_cam = _build_frustum_points_cam(
            grid_size=int(self.config.frustum_grid_size),
            depths_m=list(self.config.frustum_depths_m),
            fov_deg=float(self.config.frustum_fov_deg),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("_frustum_points_cam", frustum_points_cam, persistent=False)

        # Head input dim is data-dependent (feature channel count depends on EVL cfg).
        self.head = self.config.head.setup_target(in_dim=None)
        # Keep the trainable head modules on the same device as the frozen backbone.
        # (EvlBackbone is not an nn.Module, so nn.Module.to() won't affect it.)
        self.to(self.backbone.device)

    def _freeze_inactive_pose_encoder(self) -> None:
        """Disable gradients for the pose encoder not used by `pose_encoding_mode`."""

        match str(self.config.pose_encoding_mode):
            case "shell_sh":
                self.pose_encoder_lff.eval()
                for param in self.pose_encoder_lff.parameters():
                    param.requires_grad = False
            case "lff6d":
                self.pose_encoder_sh.eval()
                for param in self.pose_encoder_sh.parameters():
                    param.requires_grad = False
            case other:
                raise ValueError(f"Unsupported pose_encoding_mode: {other}")

    def _pool_global(self, field: Tensor) -> Tensor:
        """Pool global context from a voxel field (coarse tokens + attention pooling)."""

        if self.global_attn is None or self.global_query is None:
            raise RuntimeError("Global pooling requested but global attention modules were not initialized.")

        batch_size = int(field.shape[0])
        grid = int(self.global_token_grid_size)

        token_grid = torch.nn.functional.adaptive_avg_pool3d(field, output_size=(grid, grid, grid))
        tokens = token_grid.flatten(2).transpose(1, 2)  # B T C

        if (
            self.global_use_positional_encoding
            and self.global_pos_mlp is not None
            and self._global_token_pos.numel() > 0
        ):
            pos = self._global_token_pos.to(device=tokens.device, dtype=tokens.dtype)
            tokens = tokens + self.global_pos_mlp(pos).unsqueeze(0)  # 1 T C

        q = self.global_query.unsqueeze(0).expand(batch_size, -1, -1)  # B Q C
        pooled, _ = self.global_attn(q, tokens, tokens)
        return pooled.flatten(1)

    def _frustum_points_world(self, poses_world_cam: PoseTW) -> Tensor:
        """Generate frustum sample points in world coordinates for each candidate."""

        poses = poses_world_cam
        if poses.ndim == 2:
            poses = PoseTW(poses._data.unsqueeze(0))
        pts_cam = self._frustum_points_cam.to(device=poses.t.device, dtype=torch.float32)  # K 3

        batch_size = int(poses.t.shape[0])
        num_candidates = int(poses.t.shape[1])
        pts_cam = pts_cam.view(1, 1, -1, 3).expand(batch_size, num_candidates, -1, 3)  # B N K 3
        return poses * pts_cam

    def _pool_candidates(self, *, tokens: Tensor, valid: Tensor, pose_embed: Tensor) -> Tensor:
        """Cross-attention pooling over candidate-local frustum samples."""

        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens shape (B,N,K,C), got {tuple(tokens.shape)}.")
        if valid.shape != tokens.shape[:3]:
            raise ValueError(f"Expected valid shape {tuple(tokens.shape[:3])}, got {tuple(valid.shape)}.")

        batch_size, num_candidates, num_points, embed_dim = tokens.shape
        batch_candidates = int(batch_size) * int(num_candidates)

        q = self.cand_pose_to_query(pose_embed).reshape(batch_candidates, 1, embed_dim)
        kv = tokens.reshape(batch_candidates, num_points, embed_dim)
        key_padding_mask = (~valid).reshape(batch_candidates, num_points)  # True = ignore token

        # NOTE: torch MultiheadAttention produces NaNs when *all* keys are masked for an item.
        # We disable masking for all-invalid candidates and rely on kv being zeros. We also
        # zero the output again afterwards.
        any_valid = (~key_padding_mask).any(dim=-1, keepdim=True)  # (B*N,1)
        key_padding_mask_safe = torch.where(any_valid, key_padding_mask, torch.zeros_like(key_padding_mask))

        kv = kv * (~key_padding_mask).to(dtype=kv.dtype).unsqueeze(-1)
        pooled, _ = self.cand_attn(q, kv, kv, key_padding_mask=key_padding_mask_safe)
        pooled = pooled.squeeze(1)
        pooled = pooled * any_valid.to(dtype=pooled.dtype)
        return pooled.reshape(batch_size, num_candidates, embed_dim)

    def _get_reference_pose_world_rig(self, efm: Mapping[str, Any]) -> PoseTW:
        pose_tw = efm.get(_first_key(ARIA_POSE_T_WORLD_RIG))
        if not isinstance(pose_tw, PoseTW):
            raise KeyError(f"Missing {ARIA_POSE_T_WORLD_RIG} PoseTW in efm snippet.")
        return PoseTW.from_matrix3x4(pose_tw.matrix3x4[..., -1, :, :])

    @staticmethod
    def _ensure_candidate_batch(candidate_poses_world_cam: PoseTW) -> PoseTW:
        if candidate_poses_world_cam.ndim == 2:  # N x 12
            return PoseTW(candidate_poses_world_cam._data.unsqueeze(0))
        return candidate_poses_world_cam

    def forward(
        self,
        efm: Mapping[str, Any],
        candidate_poses_world_cam: PoseTW | None = None,
        *,
        reference_pose_world_rig: PoseTW | None = None,
        candidate_poses_camera_rig: PoseTW | None = None,
    ) -> VinPrediction:
        """Score candidate poses for one snippet.

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: Optional candidate camera poses as world←camera.
                Shape can be ``(N,12)`` or ``(B,N,12)``. If omitted, the poses are
                constructed from `candidate_poses_camera_rig` and `reference_pose_world_rig`.
            reference_pose_world_rig: Optional override for reference rig pose (world←rig).
                If omitted, uses the last pose in ``pose/t_world_rig`` from the snippet.
            candidate_poses_camera_rig: Optional candidate poses in the **reference rig frame**
                as camera←rig. If provided, pose descriptors are derived from this tensor
                (recommended for training when available, e.g. from
                ``OracleRriLabelBatch.depths.camera.T_camera_rig``).

        Returns:
            :class:`VinPrediction` with CORAL logits and expected scores.
        """

        backbone_out = self.backbone.forward(efm)
        device = backbone_out.voxel_extent.device

        if reference_pose_world_rig is None:
            pose_world_rig_ref = self._get_reference_pose_world_rig(efm).to(device=device)  # type: ignore[arg-type]
        else:
            pose_world_rig_ref = reference_pose_world_rig.to(device=device)  # type: ignore[arg-type]
        if pose_world_rig_ref.ndim == 1:
            pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.unsqueeze(0))

        # ------------------------------------------------------------------ candidate poses + relative pose
        if candidate_poses_camera_rig is not None:
            # Training-time contract: candidate poses as camera<-rig_ref.
            pose_cam_rig = self._ensure_candidate_batch(candidate_poses_camera_rig).to(device=device)  # type: ignore[arg-type]
            batch_size, num_candidates = int(pose_cam_rig.shape[0]), int(pose_cam_rig.shape[1])

            if pose_world_rig_ref.shape[0] == 1 and batch_size > 1:
                pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.expand(batch_size, 12))
            elif pose_world_rig_ref.shape[0] != batch_size:
                raise ValueError("reference_pose_world_rig must have batch size 1 or match candidate batch size.")

            pose_rig_cam = pose_cam_rig.inverse()  # rig_ref <- cam
            pose_world_cam = pose_world_rig_ref[:, None] @ pose_rig_cam  # world <- cam
        else:
            # Inference-time contract: candidate poses as world<-cam.
            if candidate_poses_world_cam is None:
                raise ValueError("candidate_poses_world_cam must be provided when candidate_poses_camera_rig is None.")

            pose_world_cam = self._ensure_candidate_batch(candidate_poses_world_cam).to(device=device)  # type: ignore[arg-type]
            batch_size, num_candidates = int(pose_world_cam.shape[0]), int(pose_world_cam.shape[1])

            if pose_world_rig_ref.shape[0] == 1 and batch_size > 1:
                pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.expand(batch_size, 12))
            elif pose_world_rig_ref.shape[0] != batch_size:
                raise ValueError("reference_pose_world_rig must have batch size 1 or match candidate batch size.")

            pose_rig_cam = pose_world_rig_ref.inverse()[:, None] @ pose_world_cam  # rig_ref <- cam

        # ------------------------------------------------------------------ pose encoding (shell descriptor)
        candidate_center_rig_m = pose_rig_cam.t.to(dtype=torch.float32)  # B N 3
        candidate_radius_m = torch.linalg.vector_norm(candidate_center_rig_m, dim=-1, keepdim=True)  # B N 1
        candidate_center_dir_rig = candidate_center_rig_m / (candidate_radius_m + 1e-8)

        cam_forward_axis_cam = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        candidate_forward_dir_rig = torch.einsum(
            "...ij,j->...i", pose_rig_cam.R.to(dtype=torch.float32), cam_forward_axis_cam
        )
        candidate_forward_dir_rig = candidate_forward_dir_rig / (
            torch.linalg.vector_norm(candidate_forward_dir_rig, dim=-1, keepdim=True) + 1e-8
        )

        view_alignment = (candidate_forward_dir_rig * (-candidate_center_dir_rig)).sum(dim=-1, keepdim=True)

        # Descriptor linking `voxel/T_world_voxel` (world←voxel) to the chosen reference pose (world←rig_ref).
        # This is snippet-level context (same for all candidates) and is broadcast to (B,N,...).
        pose_world_voxel = backbone_out.t_world_voxel
        if pose_world_voxel.ndim == 1:
            pose_world_voxel = PoseTW(pose_world_voxel._data.unsqueeze(0))
        if int(pose_world_voxel.shape[0]) != int(batch_size):
            if int(pose_world_voxel.shape[0]) == 1:
                pose_world_voxel = PoseTW(pose_world_voxel._data.expand(batch_size, 12))
            else:
                raise ValueError("voxel/T_world_voxel must have batch size 1 or match candidate batch size.")

        pose_rig_voxel = pose_world_rig_ref.inverse() @ pose_world_voxel  # rig_ref <- voxel
        voxel_origin_rig_m = pose_rig_voxel.t.to(dtype=torch.float32)  # B 3
        voxel_forward_dir_rig = torch.einsum(
            "...ij,j->...i", pose_rig_voxel.R.to(dtype=torch.float32), cam_forward_axis_cam
        )
        voxel_forward_dir_rig = voxel_forward_dir_rig / (
            torch.linalg.vector_norm(voxel_forward_dir_rig, dim=-1, keepdim=True) + 1e-8
        )
        voxel_rig_link = torch.cat([voxel_origin_rig_m, voxel_forward_dir_rig], dim=-1)  # B 6
        voxel_rig_link = voxel_rig_link.unsqueeze(1).expand(batch_size, num_candidates, -1)

        pose_encoding_mode = str(self.config.pose_encoding_mode)
        match pose_encoding_mode:
            case "shell_sh":
                pose_enc = self.pose_encoder_sh(
                    candidate_center_dir_rig,
                    candidate_forward_dir_rig,
                    r=candidate_radius_m,
                    scalars=view_alignment,
                )
            case "lff6d":
                pose_descriptor_6d = torch.cat([candidate_center_rig_m, candidate_forward_dir_rig], dim=-1)
                pose_enc = self.pose_encoder_lff(pose_descriptor_6d)
            case other:
                raise ValueError(f"Unsupported pose_encoding_mode: {other}")

        # ------------------------------------------------------------------ build voxel-aligned scene field
        field = _build_scene_field(
            backbone_out,
            use_channels=list(self.config.scene_field_channels),
            occ_input_threshold=float(self.config.occ_input_threshold),
            counts_norm_mode=self.config.counts_norm_mode,
            occ_pr_is_logits=bool(self.config.occ_pr_is_logits),
        ).to(device=device)
        field = self.field_proj(field)

        # ------------------------------------------------------------------ global pooling (coarse tokens)
        parts: list[Tensor] = [
            pose_enc.to(device=device, dtype=field.dtype),
            voxel_rig_link.to(device=device, dtype=field.dtype),
        ]
        if self.config.use_global_pool:
            global_embed = self._pool_global(field)
            parts.append(global_embed.unsqueeze(1).expand(batch_size, num_candidates, -1))

        # ------------------------------------------------------------------ candidate-conditioned frustum query
        points_world = self._frustum_points_world(pose_world_cam)
        tokens, token_valid = _sample_voxel_field(
            field,
            points_world=points_world,
            t_world_voxel=backbone_out.t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        cand_embed = self._pool_candidates(
            tokens=tokens, valid=token_valid, pose_embed=pose_enc.to(dtype=torch.float32)
        )
        parts.append(cand_embed.to(dtype=field.dtype))

        # NOTE: Candidate validity is currently defined by whether *any frustum sample point* falls inside the EVL
        # voxel grid (after mapping WORLD→VOXEL using `voxel/T_world_voxel`).
        # TODO: Consider additionally AND-ing with "camera center inside voxel grid" validity if you want to keep the
        # baseline semantics and avoid edge cases where the center is outside but shallow frustum samples still land
        # inside the grid.
        candidate_valid = token_valid.any(dim=-1)

        feats = torch.cat(parts, dim=-1)
        feats = feats * candidate_valid.to(dtype=feats.dtype).unsqueeze(-1)
        logits = self.head(feats.reshape(batch_size * num_candidates, -1)).reshape(batch_size, num_candidates, -1)

        prob = coral_logits_to_prob(logits)
        expected, expected_norm = coral_expected_from_logits(logits)

        return VinPrediction(
            logits=logits,
            prob=prob,
            expected=expected,
            expected_normalized=expected_norm,
            candidate_valid=candidate_valid,
        )

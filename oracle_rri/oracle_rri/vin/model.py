"""VIN model on top of a frozen EVL backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from efm3d.aria.aria_constants import ARIA_CALIB, ARIA_POSE_T_WORLD_RIG
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field, field_validator
from torch import Tensor, nn

from ..utils import BaseConfig
from .backbone_evl import EvlBackboneConfig
from .coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob
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
    # NOTE: For v0.1 we use a simple pinhole-style directional grid defined by a single
    # symmetric FOV prior. This intentionally ignores per-snippet intrinsics; replace with
    # intrinsics-aware unprojection if needed later.
    s = math.tan(math.radians(fov_deg) / 2.0)
    xy = torch.linspace(-s, s, steps=grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")

    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)  # G 3
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    depths = torch.tensor(depths_m, device=device, dtype=dtype).reshape(-1, 1)  # D 1
    points = (dirs.unsqueeze(0) * depths.unsqueeze(1)).reshape(-1, 3)  # (D*G) 3
    return points


def _build_frustum_points_cam_from_camera(
    camera: CameraTW,
    *,
    grid_size: int,
    depths_m: list[float],
) -> Tensor:
    cam = camera
    if cam.ndim == 2:
        cam = cam[-1]
    elif cam.ndim == 3:
        cam = cam[0, -1]
    if cam.ndim != 1:
        raise ValueError(f"Expected CameraTW with ndim=1 after slicing, got {cam.ndim}.")

    width, height = cam.size.to(dtype=torch.float32)
    cx, cy = cam.c.to(dtype=torch.float32)
    valid_radius = cam.valid_radius.to(dtype=torch.float32)

    width_f = float(width.item())
    height_f = float(height.item())
    cx_f = float(cx.item())
    cy_f = float(cy.item())

    if (
        valid_radius.numel() == 2
        and torch.isfinite(valid_radius).all()
        and float(valid_radius.min().item()) > 0.0
        and math.isfinite(width_f)
        and math.isfinite(height_f)
        and width_f > 1.0
        and height_f > 1.0
    ):
        radius_x = 0.95 * min(float(valid_radius[0].item()), 0.5 * (width_f - 1.0))
        radius_y = 0.95 * min(float(valid_radius[1].item()), 0.5 * (height_f - 1.0))
    else:
        # Conservative fallback: use a centered grid within the image bounds.
        radius_x = 0.95 * max(1.0, 0.5 * (width_f - 1.0))
        radius_y = 0.95 * max(1.0, 0.5 * (height_f - 1.0))

    x_min = max(0.0, cx_f - radius_x)
    x_max = min(width_f - 1.0, cx_f + radius_x)
    y_min = max(0.0, cy_f - radius_y)
    y_max = min(height_f - 1.0, cy_f + radius_y)

    xs = torch.linspace(x_min, x_max, steps=grid_size, device=cam.device, dtype=torch.float32)
    ys = torch.linspace(y_min, y_max, steps=grid_size, device=cam.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    p2d = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    rays, valid = cam.unproject(p2d.unsqueeze(0))
    rays = rays.squeeze(0)
    valid = valid.squeeze(0)
    rays = rays / (torch.linalg.vector_norm(rays, dim=-1, keepdim=True) + 1e-8)
    rays = torch.where(valid.unsqueeze(-1), rays, torch.nan)

    depths = torch.tensor(depths_m, device=cam.device, dtype=cam.dtype).reshape(-1, 1)  # D 1
    points = (rays.unsqueeze(0) * depths.unsqueeze(1)).reshape(-1, 3)  # (D*G) 3
    return points


def _safe_mean_pool(tokens: Tensor, valid: Tensor) -> Tensor:
    """Mean pool tokens with a validity mask over the sample dimension."""

    if tokens.ndim != 4:
        raise ValueError(f"Expected tokens shape (B,N,K,C), got {tuple(tokens.shape)}.")
    if valid.shape != tokens.shape[:3]:
        raise ValueError(f"Expected valid shape {tuple(tokens.shape[:3])}, got {tuple(valid.shape)}.")
    mask = valid.to(dtype=tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=-2).clamp_min(1.0)
    pooled = (tokens * mask).sum(dim=-2) / denom
    return pooled


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
            occ_evidence = (parts["occ_input"] > occ_input_threshold).to(dtype=torch.float32)
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
        if counts_norm_mode == "log1p":
            parts["counts_norm"] = torch.log1p(counts).unsqueeze(1) / torch.log1p(max_counts).unsqueeze(1)
        else:  # "linear"
            parts["counts_norm"] = (counts / max_counts).unsqueeze(1)

    if "new_surface_prior" in use_channels:
        parts["new_surface_prior"] = parts["unknown"] * parts["occ_pr"]

    field_parts: list[Tensor] = []
    for name in use_channels:
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


def _candidate_valid_from_token(token_valid: Tensor, *, min_valid_frac: float) -> Tensor:
    """Convert frustum token validity into a per-candidate mask."""

    if token_valid.ndim < 1:
        raise ValueError(f"Expected token_valid with ndim>=1, got {tuple(token_valid.shape)}.")
    valid_frac = token_valid.float().mean(dim=-1)
    return valid_frac >= min_valid_frac


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
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()

        act: nn.Module
        match activation:
            case "relu":
                act = nn.ReLU()
            case "gelu":
                act = nn.GELU()

        layers: list[nn.Module] = []
        if in_dim is None:
            layers.append(nn.LazyLinear(hidden_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)
        self.coral = CoralLayer(in_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.coral(self.mlp(x))


class VinScorerHeadConfig(BaseConfig[VinScorerHead]):
    """Configuration for :class:`VinScorerHead`."""

    target: type[VinScorerHead] = Field(default_factory=lambda: VinScorerHead, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    hidden_dim: int = Field(default=128, gt=0)
    """Hidden dimension for MLP layers."""

    num_layers: int = Field(default=1, ge=1)
    """Number of MLP layers before the CORAL layer."""

    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Dropout probability in the MLP."""

    num_classes: int = Field(default=15, ge=2)
    """Number of ordinal bins (VIN-NBV uses 15)."""

    activation: Literal["gelu", "relu"] = "gelu"
    """Activation function ('gelu' or 'relu')."""

    def setup_target(self, *, in_dim: int | None = None) -> VinScorerHead:  # type: ignore[override]
        return self.target(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
            activation=self.activation,
        )


def _vin_target() -> type["VinModel"]:
    return VinModel


class VinModelConfig(BaseConfig["VinModel"]):
    """Configuration for :class:`VinModel`."""

    target: type["VinModel"] = Field(default_factory=_vin_target, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoder_sh: ShellShPoseEncoderConfig = Field(default_factory=ShellShPoseEncoderConfig)
    """Spherical harmonics pose encoding configuration (shell descriptor)."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: list[str] = Field(
        default_factory=lambda: ["occ_pr", "occ_input", "counts_norm"],
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

    frustum_fov_deg: float = Field(default=90.0, gt=0.0, lt=180.0)
    """Approximate symmetric FOV used for the candidate frustum sampling grid."""

    use_global_pool: bool = True
    """Whether to concatenate the global mean-pooled embedding to per-candidate features."""

    candidate_min_valid_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    """Minimum fraction of valid frustum samples required to keep a candidate."""

    @field_validator("scene_field_channels")
    @classmethod
    def _validate_scene_field_channels(cls, value: list[str]) -> list[str]:
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
        bad = [d for d in value if (not math.isfinite(d)) or d <= 0.0]
        if bad:
            raise ValueError(f"frustum_depths_m must contain finite values > 0, got {bad}")
        return value


class VinModel(nn.Module):
    """View Introspection Network (VIN) predicting RRI from EVL voxel features + candidate pose."""

    def __init__(self, config: VinModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = self.config.backbone.setup_target()
        self.pose_encoder_sh = self.config.pose_encoder_sh.setup_target()

        field_dim = self.config.field_dim
        gn_groups = _largest_divisor_leq(field_dim, self.config.field_gn_groups)

        self.field_proj = nn.Sequential(
            nn.LazyConv3d(field_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )

        self.use_global_pool = self.config.use_global_pool

        # NOTE: Register the canonical frustum sample points as a buffer so it moves with `.to(...)`.
        # The points are deterministic (derived from config), hence `persistent=False`.
        self._frustum_points_cam: Tensor
        frustum_points_cam = _build_frustum_points_cam(
            grid_size=self.config.frustum_grid_size,
            depths_m=self.config.frustum_depths_m,
            fov_deg=self.config.frustum_fov_deg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("_frustum_points_cam", frustum_points_cam, persistent=False)

        # Head input dim is data-dependent (feature channel count depends on EVL cfg).
        self.head = self.config.head.setup_target(in_dim=None)
        # Keep the trainable head modules on the same device as the frozen backbone.
        # (EvlBackbone is not an nn.Module, so nn.Module.to() won't affect it.)
        self.to(self.backbone.device)

    def _pool_global(self, field: Tensor) -> Tensor:
        """Mean-pool global context from a voxel field."""

        return field.mean(dim=(-3, -2, -1))

    def _frustum_points_world(self, poses_world_cam: PoseTW, *, camera: CameraTW | None) -> Tensor:
        """Generate frustum sample points in world coordinates for each candidate."""

        poses = poses_world_cam
        if poses.ndim == 2:
            poses = PoseTW(poses._data.unsqueeze(0))
        if camera is None:
            pts_cam = self._frustum_points_cam.to(device=poses.t.device, dtype=torch.float32)  # K 3
        else:
            cam = camera.to(device=poses.t.device, dtype=torch.float32)
            try:
                pts_cam = _build_frustum_points_cam_from_camera(
                    cam,
                    grid_size=self.config.frustum_grid_size,
                    depths_m=self.config.frustum_depths_m,
                )
            except (RuntimeError, ValueError):
                # CameraTW.unproject only supports some camera models (e.g., fisheye624, pinhole).
                # Fall back to the fixed-FOV grid to keep inference/training robust.
                pts_cam = self._frustum_points_cam.to(device=poses.t.device, dtype=torch.float32)  # K 3

        batch_size = int(poses.t.shape[0])
        num_candidates = int(poses.t.shape[1])
        pts_cam = pts_cam.view(1, 1, -1, 3).expand(batch_size, num_candidates, -1, 3)  # B N K 3
        return poses * pts_cam

    def _pool_candidates(self, *, tokens: Tensor, valid: Tensor) -> Tensor:
        """Mean-pool candidate-local frustum samples."""

        return _safe_mean_pool(tokens, valid)

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

        pose_enc = self.pose_encoder_sh(
            candidate_center_dir_rig,
            candidate_forward_dir_rig,
            r=candidate_radius_m,
            scalars=view_alignment,
        )

        # ------------------------------------------------------------------ build voxel-aligned scene field
        field = _build_scene_field(
            backbone_out,
            use_channels=self.config.scene_field_channels,
            occ_input_threshold=self.config.occ_input_threshold,
            counts_norm_mode=self.config.counts_norm_mode,
            occ_pr_is_logits=self.config.occ_pr_is_logits,
        ).to(device=device)
        field = self.field_proj(field)

        # ------------------------------------------------------------------ global pooling (coarse tokens)
        parts: list[Tensor] = [pose_enc.to(device=device, dtype=field.dtype)]
        if self.use_global_pool:
            global_embed = self._pool_global(field)
            parts.append(global_embed.unsqueeze(1).expand(batch_size, num_candidates, -1))

        # ------------------------------------------------------------------ candidate-conditioned frustum query
        camera = efm.get(_first_key(ARIA_CALIB))
        points_world = self._frustum_points_world(
            pose_world_cam,
            camera=camera if isinstance(camera, CameraTW) else None,
        )
        tokens, token_valid = _sample_voxel_field(
            field,
            points_world=points_world,
            t_world_voxel=backbone_out.t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        cand_embed = self._pool_candidates(tokens=tokens, valid=token_valid)
        parts.append(cand_embed.to(dtype=field.dtype))

        # NOTE: Candidate validity is based on the fraction of frustum samples that fall inside the EVL voxel grid
        # (after mapping WORLD→VOXEL using `voxel/T_world_voxel`). This avoids admitting candidates with only a
        # handful of in-bounds samples.
        candidate_valid = _candidate_valid_from_token(
            token_valid,
            min_valid_frac=self.config.candidate_min_valid_frac,
        )

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

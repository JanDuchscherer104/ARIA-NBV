"""VIN v3 (core) for RRI prediction with evidence-backed components.

This module implements a streamlined VIN baseline distilled from the vin-v2
optuna sweep. The sweep showed that extra modules (PointNeXt point encoder,
semidense frustum MHCA, trajectory context) were confounded by config fixes or
only weakly positive. The most reliable signal came from semidense projection
coverage and voxel-validity cues, so v3 keeps a minimal, deterministic path:

1) Pose encoding (R6D + LFF):
   Candidate poses are expressed in the reference rig frame
   T_rig_ref_cam = T_world_rig_ref^{-1} * T_world_cam and encoded as translation
   plus rotation-6D with Learnable Fourier Features.

2) Scene field (fixed channels):
   The voxel field concatenates occ_pr, cent_pr, counts_norm, occ_input,
   free_input, and new_surface_prior. We normalize counts as
   counts_norm = log1p(n) / log1p(max(n)) and define unknown = 1 - counts_norm,
   new_surface_prior = unknown * occ_pr. This compact field was stable in sweep
   diagnostics and supports voxel-validity gating.

3) Global context (pose-conditioned attention):
   A pooled voxel grid is attended by pose embeddings, with LFF positional keys
   in the reference rig frame.

4) Semidense projection stats (VIN-NBV proxy):
   We project semidense points into each candidate view to compute coverage,
   empty fraction, visibility fraction, and depth moments. These features act as
   a lightweight proxy for frustum attention and are used for candidate validity
   and diagnostics (no FiLM, no concat).

5) Voxel projection FiLM:
   Pooled voxel centers are projected into candidate views and summarized; this
   drives a light FiLM modulation of the global feature (kept as the only
   view-conditioned modulation).

6) CORAL head:
   A shallow MLP plus CORAL ordinal head produces per-candidate RRI scores.

Frame-consistency:
Candidate generation applies rotate_yaw_cw90 (a local +Z roll) to poses for UI
alignment. EVL backbone outputs do not use this convention. VinModelV3 therefore
undoes this rotation before computing pose features. If apply_cw90_correction is
enabled, callers must pre-correct p3d_cameras and set cw90_corrected=True.

NOTE: vin inputs are typically VinSnippetView with points_world shaped (N,5)
containing (x, y, z, 1/sigma_d, n_obs). This file enforces that contract to
avoid silent failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator
from pytorch3d.renderer.cameras import (  # type: ignore[import-untyped]
    PerspectiveCameras,
)
from torch import Tensor, nn
from torch.nn import functional as functional

from oracle_rri.utils.frames import rotate_yaw_cw90

from ..data.efm_views import EfmSnippetView, VinSnippetView
from ..data.vin_snippet_utils import build_vin_snippet_view
from ..rri_metrics.coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob
from ..utils import BaseConfig
from .backbone_evl import EvlBackboneConfig
from .pose_encoders import PoseEncoder, R6dLffPoseEncoderConfig
from .pose_encoding import LearnableFourierFeaturesConfig
from .summarize_v3 import summarize_vin_v3
from .types import EvlBackboneOutput, VinPrediction, VinV3ForwardDiagnostics
from .vin_modules import PoseConditionedGlobalPool
from .vin_utils import (
    FieldBundle,
    GlobalContext,
    PoseFeatures,
    PreparedInputs,
    ensure_candidate_batch,
    ensure_pose_batch,
    largest_divisor_leq,
    pos_grid_from_pts_world,
    sample_voxel_field,
)

if TYPE_CHECKING:
    from oracle_rri.data.vin_oracle_types import VinOracleBatch

    from .pose_encoding import LearnableFourierFeatures


FIELD_CHANNELS_V3: tuple[str, ...] = (
    "occ_pr",
    "occ_input",
    "counts_norm",
    "cent_pr",
    "free_input",
    "unknown",
    "new_surface_prior",
)

SEMIDENSE_PROJ_FEATURES: tuple[str, ...] = (
    "coverage",
    "empty_frac",
    "semidense_candidate_vis_frac",
    "depth_mean",
    "depth_std",
)
SEMIDENSE_PROJ_DIM = len(SEMIDENSE_PROJ_FEATURES)

SEMIDENSE_PROJ_FEATURE_ALIASES: dict[str, str] = {
    "valid_frac": "semidense_candidate_vis_frac",
    "semidense_valid_frac": "semidense_candidate_vis_frac",
}


class VinModelV3Config(BaseConfig["VinModelV3"]):
    """Configuration for :class:`VinModelV3` (streamlined VIN baseline)."""

    target: type["VinModelV3"] = Field(default_factory=lambda: VinModelV3, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target` (config-as-factory)."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration that supplies voxel features."""

    pose_encoder: R6dLffPoseEncoderConfig = Field(default_factory=R6dLffPoseEncoderConfig)
    """Pose encoder configuration (R6D + LFF; stable relative pose encoding)."""

    pos_grid_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(
            input_dim=3,
            fourier_dim=32,
            hidden_dim=32,
            output_dim=16,
        ),
    )
    """LFF encoder for XYZ voxel position keys used by global pooling."""

    head_hidden_dim: int = Field(default=192, gt=0)
    """Hidden dimension for the scorer MLP (optuna favored compact heads)."""

    head_num_layers: int = Field(default=1, ge=1)
    """Number of MLP layers before the CORAL layer (best trials used 1)."""

    head_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    """Dropout probability in the MLP (sweep best used near-zero dropout)."""

    num_classes: int = Field(default=15, ge=2)
    """Number of ordinal bins (VIN-NBV uses 15 for sweep comparability)."""

    coral_preinit_bias: bool = True
    """Pre-initialize CORAL biases for faster, more stable ordinal learning."""

    field_dim: int = Field(default=16, gt=0)
    """Channel dimension of the projected voxel field (compact by design)."""

    field_gn_groups: int = Field(default=4, gt=0)
    """Requested GroupNorm groups (clamped to a divisor of ``field_dim``) for stability."""

    semidense_proj_grid_size: int = Field(default=16, gt=0)
    """Grid size for semidense coverage stats (sweep best used 12)."""

    semidense_proj_max_points: int = Field(default=4096, gt=0)
    """Maximum semidense points used for projection stats (sweep best used 4096)."""
    use_voxel_valid_frac_gate: bool = True
    """Gate voxel/global features based on voxel coverage (best trials often off)."""

    semidense_obs_count_min: float = 1.0
    """Global minimum of semidense observation count ``n_obs`` (cache summary)."""

    semidense_obs_count_max: float = 40.0
    """Global maximum of semidense observation count ``n_obs`` (cache summary)."""

    semidense_obs_count_p95: float = 11.0
    """Global 95th percentile of semidense observation count ``n_obs`` (cache summary)."""

    semidense_obs_count_mean: float = 4.3714
    """Global mean of semidense observation count ``n_obs`` (cache summary)."""

    semidense_obs_count_std: float = 3.3134
    """Global standard deviation of semidense observation count ``n_obs`` (cache summary)."""

    semidense_inv_dist_std_min: float = 0.0
    """Global minimum of semidense inverse depth std ``1/sigma_d`` (cache summary)."""

    semidense_inv_dist_std_max: float = 0.03
    """Global maximum of semidense inverse depth std ``1/sigma_d`` (cache summary)."""

    semidense_inv_dist_std_p95: float = 0.011
    """Global 95th percentile of semidense inverse depth std ``1/sigma_d`` (cache summary)."""

    semidense_inv_dist_std_mean: float = 0.0032
    """Global mean of semidense inverse depth std ``1/sigma_d`` (cache summary)."""

    semidense_inv_dist_std_std: float = 0.0040
    """Global standard deviation of semidense inverse depth std ``1/sigma_d`` (cache summary)."""

    apply_cw90_correction: bool = False
    """Undo ``rotate_yaw_cw90`` on poses (requires CW90-corrected cameras)."""

    global_pool_grid_size: int = Field(default=6, gt=0)
    """Target grid size for pose-conditioned global pooling (best trials used ~5)."""

    scene_field_channels: list[
        Literal[
            "occ_pr",
            "occ_input",
            "counts_norm",
            "observed",
            "unknown",
            "free_input",
            "cent_pr",
            "new_surface_prior",
        ]
    ] = Field(
        default_factory=lambda: [
            *FIELD_CHANNELS_V3,
        ],
    )

    """Ordered list of scene-field channels to include in the voxel field.

    This keeps the voxel representation compact and aligned with the sweep
    evidence favoring coverage- and prior-aware features.
    """

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

    # NOTE: No additional model validators; VIN-Core keeps a fixed surface area.


class VinModelV3(nn.Module):
    """VIN-Core head for RRI prediction with a minimal evidence-backed feature set.

    The vin-v2 optuna sweep showed weak or confounded gains for heavy modules
    (PointNeXt point encoder, frustum MHCA, trajectory context). VIN v3 therefore
    focuses on pose encoding, compact voxel evidence, and semidense projection
    stats, while enforcing fail-fast contracts to avoid silent collapse.
    """

    def __init__(self, config: VinModelV3Config) -> None:
        super().__init__()
        self.config = config

        # Optional modules (may be None)
        self.voxel_gate: nn.Module | None = None
        self.voxel_proj_film: nn.Module | None = None
        self.voxel_proj_film_norm: nn.GroupNorm | None = None

        # Init backbone lazily during first forward pass iff backbone outputs are not provided
        self.backbone = None

        self.pose_encoder: PoseEncoder = self.config.pose_encoder.setup_target()

        field_dim = self.config.field_dim
        gn_groups = largest_divisor_leq(field_dim, self.config.field_gn_groups)
        self.field_proj = nn.Sequential(
            nn.Conv3d(
                len(self.config.scene_field_channels),
                field_dim,
                kernel_size=1,
                bias=False,  # embedding-like projection
            ),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )

        pose_dim = self.pose_encoder.out_dim
        num_heads = largest_divisor_leq(field_dim, 4)
        self.global_pooler = PoseConditionedGlobalPool(
            field_dim=field_dim,
            pose_dim=pose_dim,
            pool_size=self.config.global_pool_grid_size,
            num_heads=num_heads,
            pos_grid_encoder=self.config.pos_grid_encoder_lff,
        )
        # TODO: why are we using sigmoid gate as well as FiLM?
        if self.config.use_voxel_valid_frac_gate:
            self.voxel_gate = nn.Sequential(
                nn.Linear(1, field_dim),
                nn.Sigmoid(),
            )

        self.voxel_proj_film = nn.Linear(SEMIDENSE_PROJ_DIM, 2 * field_dim, bias=True)
        voxel_proj_groups = largest_divisor_leq(field_dim, 4)
        self.voxel_proj_film_norm = nn.GroupNorm(
            num_groups=voxel_proj_groups,
            num_channels=field_dim,
        )

        # ---------------------------------------------------------------------------------
        # Scorer head: MLP + CORAL
        head_in_dim = pose_dim + field_dim
        act: nn.Module = nn.GELU()
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

        self.to(self.backbone.device if self.backbone is not None else torch.device("cpu"))

    @property
    def pose_encoder_lff(self) -> LearnableFourierFeatures | None:
        """Return the LFF sub-encoder when present (else ``None``).

        Useful for diagnostics and consistency checks when pose encoding is a
        critical signal in the streamlined baseline.
        """
        return getattr(self.pose_encoder, "pose_encoder_lff", None)

    def _ensure_vin_snippet(
        self,
        efm: EfmSnippetView | VinSnippetView,
        *,
        device: torch.device,
    ) -> VinSnippetView:
        """Ensure a VinSnippetView is available for semidense projection stats.

        VIN v3 consumes padded semidense points, so we always operate on
        :class:`VinSnippetView`. Full EFM snippets are converted on demand.

        Args:
            efm (EfmSnippetView | VinSnippetView): EFM/VIN snippet.
            device (torch.device): Target device.

        Returns:
            VinSnippetView: Padded snippet with points_world (Tensor["B, P, C_sem"]).
        """
        if isinstance(efm, VinSnippetView):
            return efm.to(device=device)
        if isinstance(efm, EfmSnippetView):
            return build_vin_snippet_view(
                efm,
                device=device,
                max_points=self.config.semidense_proj_max_points,
                include_inv_dist_std=True,
                include_obs_count=True,
            )
        raise TypeError(
            "VinModelV3 expects a VinSnippetView or EfmSnippetView for `efm`.",
        )

    def _prepare_inputs(
        self,
        snippet: VinSnippetView,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        backbone_out: EvlBackboneOutput,
    ) -> PreparedInputs:
        """Prepare batched inputs, align frames, and enforce required inputs.

        This normalizes candidate/reference pose shapes, applies CW90 undo when
        requested, and verifies that semidense snippet data is present to avoid
        the silent mode-collapse observed when projection stats are missing.

        Args:
            snippet (VinSnippetView): VIN snippet with padded semidense points.
                - points_world (Tensor["B, P, C_sem"]): XYZ + extras.
                - lengths (Tensor["B"]): valid point counts per batch item.
                - t_world_rig (PoseTW["B, T, 12"]): rig trajectory poses.
            candidate_poses_world_cam (PoseTW["B, Nq, 12"]): Candidate camera poses T_w^c.
                Unbatched inputs (Nq, 12) are expanded to B=1.
            reference_pose_world_rig (PoseTW["B, 12"]): Reference rig pose T_w^r.
            backbone_out (EvlBackboneOutput): Backbone outputs (device + voxel frame).
                - t_world_voxel (PoseTW["B, 12"]): voxel grid pose in world frame.

        Returns:
            PreparedInputs:
                pose_world_cam (PoseTW["B, Nq, 12"]): Candidate poses in world frame.
                pose_world_rig_ref (PoseTW["B, 12"]): Reference rig pose in world frame.
                t_world_voxel (PoseTW["B, 12"]): Voxel frame pose in world frame.
                batch_size (int): B (batch size).
                num_candidates (int): Nq (number of candidates).
                device (torch.device): Device for downstream tensors.
                snippet (VinSnippetView): Original snippet (padded semidense).
        """
        device = backbone_out.voxel_extent.device
        pose_world_cam = ensure_candidate_batch(candidate_poses_world_cam).to(
            device=device,
        )
        batch_size, num_candidates = (
            int(pose_world_cam.shape[0]),
            int(pose_world_cam.shape[1]),
        )
        pose_world_rig_ref = ensure_pose_batch(
            reference_pose_world_rig.to(device=device),
            batch_size=batch_size,
            name="reference_pose_world_rig",
        )
        if self.config.apply_cw90_correction:
            pose_world_cam = rotate_yaw_cw90(pose_world_cam, undo=True)
            pose_world_rig_ref = rotate_yaw_cw90(pose_world_rig_ref, undo=True)

        t_world_voxel = ensure_pose_batch(
            backbone_out.t_world_voxel,
            batch_size=batch_size,
            name="voxel/T_world_voxel",
        )
        return PreparedInputs(
            pose_world_cam=pose_world_cam,
            pose_world_rig_ref=pose_world_rig_ref,
            t_world_voxel=t_world_voxel,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=device,
            snippet=snippet,
        )

    def _encode_pose_features(
        self,
        pose_world_cam: PoseTW,
        pose_world_rig_ref: PoseTW,
    ) -> PoseFeatures:
        """Encode candidate poses in the reference rig frame.

        Args:
            pose_world_cam (PoseTW["B, Nq, 12"]): SE(3) candidate camera poses in world frame, T_w^c.
            pose_world_rig_ref (PoseTW["B, 12"]): SE(3) reference rig pose in world frame, T_w^r.

        Returns:
            PoseFeatures (dataclass):
                pose_enc (Tensor["B, Nq, F_pose"]): Encoded pose features.
                pose_vec (Tensor["B, Nq, 9"]): Pose vector (t + R6D).
                candidate_center_rig_m (Tensor["B, Nq, 3"]): Candidate centers in rig frame (meters).

        Relative pose encoding (R6D + LFF) avoids global-frame drift and was consistently stable in the vin-v2 sweep.
        """
        pose_rig_cam = pose_world_rig_ref.inverse()[:, None] @ pose_world_cam
        pose_out = self.pose_encoder.encode(pose_rig_cam)
        return PoseFeatures(
            pose_enc=pose_out.pose_enc,
            pose_vec=pose_out.pose_vec,
            candidate_center_rig_m=pose_out.center_m,
        )

    def _build_field_bundle(self, backbone_out: EvlBackboneOutput) -> FieldBundle:
        """Construct the compact voxel scene field and its projection.

        The counts_norm/unknown/new_surface_prior channels encode coverage and
        surface priors that proved robust in sweep diagnostics, while keeping
        the field low-dimensional for stability.

        Args:
            backbone_out (EvlBackboneOutput): Backbone voxel outputs.
                - occ_pr (Tensor["B, 1, D, H, W"]): occupancy probability.
                - occ_input (Tensor["B, 1, D, H, W"]): observed occupancy evidence.
                - counts (Tensor["B, D, H, W"]): observation counts per voxel.
                - cent_pr (Tensor["B, 1, D, H, W"]): centerness prior.
                - free_input (Tensor["B, 1, D, H, W"], optional): free-space evidence.

        Returns:
            FieldBundle:
                field_in (Tensor["B, F_in, D, H, W"]): concatenated scene channels.
                field (Tensor["B, F_g, D, H, W"]): projected field (Conv3d + GN + GELU).
                aux (dict[str, Tensor]): named channels used for diagnostics:
                    occ_pr, cent_pr, occ_input, counts_norm, observed, unknown,
                    free_input, new_surface_prior (each shaped like a single-channel field).
        """

        if not isinstance(backbone_out.occ_pr, torch.Tensor):
            raise RuntimeError("VIN v3 requires backbone_out.occ_pr to be a Tensor.")
        if not isinstance(backbone_out.cent_pr, torch.Tensor):
            raise RuntimeError("VIN v3 requires backbone_out.cent_pr to be a Tensor.")
        if not isinstance(backbone_out.occ_input, torch.Tensor):
            raise RuntimeError("VIN v3 requires backbone_out.occ_input to be a Tensor.")
        if not isinstance(backbone_out.counts, torch.Tensor):
            raise RuntimeError("VIN v3 requires backbone_out.counts to be a Tensor.")

        occ_pr = backbone_out.occ_pr.to(dtype=torch.float32)  # type: ignore
        cent_pr = backbone_out.cent_pr.to(dtype=torch.float32)  # type: ignore
        occ_input = backbone_out.occ_input.to(dtype=torch.float32)  # type: ignore
        counts = backbone_out.counts.to(dtype=torch.float32)  # type: ignore

        max_counts = counts.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)  # B,1,1,1
        counts_norm = torch.log1p(counts) / torch.log1p(max_counts)
        counts_norm = counts_norm.unsqueeze(1).clamp(0.0, 1.0)
        observed = (counts > 0).to(dtype=counts_norm.dtype).unsqueeze(1)
        unknown = (1.0 - counts_norm).clamp(0.0, 1.0)
        if isinstance(backbone_out.free_input, torch.Tensor):
            free_input = backbone_out.free_input.to(dtype=torch.float32)
        else:
            free_input = observed * (1.0 - occ_input)
        new_surface_prior = unknown * occ_pr

        field_aux = {
            "occ_pr": occ_pr,
            "cent_pr": cent_pr,
            "occ_input": occ_input,
            "counts_norm": counts_norm,
            "observed": observed,
            "unknown": unknown,
            "free_input": free_input,
            "new_surface_prior": new_surface_prior,
        }
        missing = [name for name in self.config.scene_field_channels if name not in field_aux]
        if missing:
            raise ValueError(
                f"VinModelV3.scene_field_channels contains unknown entries: {missing}. Available: {sorted(field_aux)}.",
            )
        field_parts = [field_aux[name] for name in self.config.scene_field_channels]
        field_in = torch.cat(field_parts, dim=1)
        field_in = field_in.to(device=backbone_out.voxel_extent.device)
        field = self.field_proj(field_in)
        return FieldBundle(field_in=field_in, field=field, aux=field_aux)

    def _compute_global_context(
        self,
        field: Tensor,  # (B, F_g, D, H, W)
        pose_enc: Tensor,  # (B, Nq, F_pose)
        *,
        pts_world: Tensor,  # (B, V, 3)
        t_world_voxel: PoseTW,
        pose_world_rig_ref: PoseTW,
        voxel_extent: Tensor,
    ) -> GlobalContext:
        """Compute pose-conditioned global features from the scene field.

        A pooled voxel grid is attended by pose embeddings with LFF positional
        keys, replacing heavier frustum attention that showed weak gains.

        Args:
            field (Tensor["B, F_g, D, H, W"]): Projected voxel field.
            pose_enc (Tensor["B, Nq, F_pose"]): Pose embeddings per candidate.
            pts_world (Tensor["B, V, 3"]): Voxel center positions in world frame.
            t_world_voxel (PoseTW["B, 12"]): Voxel frame pose in world frame.
            pose_world_rig_ref (PoseTW["B, 12"]): Reference rig pose in world frame.
            voxel_extent (Tensor["B, 6"]): Voxel extent [xmin,xmax,ymin,ymax,zmin,zmax].

        Returns:
            GlobalContext:
                pos_grid (Tensor["B, 3, D, H, W"]): Normalized XYZ grid in rig_ref frame.
                global_feat (Tensor["B, Nq, F_g"]): Pose-conditioned global tokens.
        """
        # pos_grid is normalized XYZ in the reference rig frame, scaled by voxel extent.
        pos_grid = pos_grid_from_pts_world(
            pts_world.to(device=field.device, dtype=field.dtype),
            t_world_voxel=t_world_voxel,
            pose_world_rig_ref=pose_world_rig_ref,
            voxel_extent=voxel_extent,
            grid_shape=(field.shape[-3], field.shape[-2], field.shape[-1]),
        )  # B
        global_feat = self.global_pooler.forward(field, pose_enc, pos_grid=pos_grid).to(
            dtype=field.dtype,
        )
        return GlobalContext(pos_grid=pos_grid, global_feat=global_feat)

    def _pool_voxel_points(
        self,
        pts_world: Tensor,
        *,
        grid_shape: tuple[int, int, int],
        pool_grid: int,
    ) -> Tensor:
        """Downsample voxel center points to match the pooled token grid.

        This keeps voxel-projection summaries aligned with the global pool
        resolution and handles padded grids via symmetric center-cropping.

        Args:
            pts_world (Tensor["B, V, 3"] or Tensor["B, D, H, W, 3"]): Voxel centers in world frame.
            grid_shape (tuple[int, int, int]): Target (D, H, W) grid shape.
            pool_grid (int): G_pool (pooled grid size per axis).

        Returns:
            Tensor["B, P_proj, 3"]: Pooled voxel centers with P_proj = G_pool^3.
        """

        def _infer_pts_shape(num_pts: int, target_shape: tuple[int, int, int]) -> tuple[int, int, int]:
            d_t, h_t, w_t = target_shape
            if num_pts == d_t * h_t * w_t:
                return target_shape
            for pad in range(1, 4):
                d_p, h_p, w_p = d_t + 2 * pad, h_t + 2 * pad, w_t + 2 * pad
                if num_pts == d_p * h_p * w_p:
                    return (d_p, h_p, w_p)
            raise ValueError(
                "pts_world size mismatch: "
                f"got {num_pts} points; expected {d_t * h_t * w_t} "
                f"for grid_shape {target_shape} or a symmetric padding variant.",
            )

        def _center_crop(grid: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
            d0, h0, w0 = int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
            d_t, h_t, w_t = target_shape
            if (d0, h0, w0) == target_shape:
                return grid
            if d0 < d_t or h0 < h_t or w0 < w_t:
                raise ValueError(
                    f"pts_world grid {d0, h0, w0} smaller than target {target_shape}.",
                )
            if (d0 - d_t) % 2 != 0 or (h0 - h_t) % 2 != 0 or (w0 - w_t) % 2 != 0:
                raise ValueError(
                    f"pts_world grid {d0, h0, w0} cannot be center-cropped to {target_shape}.",
                )
            d_start = (d0 - d_t) // 2
            h_start = (h0 - h_t) // 2
            w_start = (w0 - w_t) // 2
            return grid[
                :,
                d_start : d_start + d_t,
                h_start : h_start + h_t,
                w_start : w_start + w_t,
                :,
            ]

        if pts_world.ndim == 3:
            batch_size, num_pts, _ = pts_world.shape
            pts_shape = _infer_pts_shape(int(num_pts), grid_shape)
            pts_grid = pts_world.view(
                batch_size,
                pts_shape[0],
                pts_shape[1],
                pts_shape[2],
                3,
            )
            pts_grid = _center_crop(pts_grid, grid_shape)
        elif pts_world.ndim == 5 and pts_world.shape[-1] == 3:
            pts_grid = _center_crop(pts_world, grid_shape)
        else:
            raise ValueError(
                f"Expected pts_world shape (B,D,H,W,3) or (B,N,3), got {tuple(pts_world.shape)}.",
            )
        grid = int(pool_grid)
        pts_grid = pts_grid.to(dtype=torch.float32).permute(0, 4, 1, 2, 3)
        pts_pool = functional.adaptive_avg_pool3d(
            pts_grid,
            output_size=(grid, grid, grid),
        )
        pts_tokens = pts_pool.flatten(2).transpose(1, 2)
        return pts_tokens

    @staticmethod
    def _apply_film(
        global_feat: Tensor,
        proj_feat: Tensor,
        *,
        film: nn.Module,
        norm: nn.GroupNorm | None,
    ) -> Tensor:
        """Apply FiLM modulation to global features.

        VIN v3 keeps only the voxel-projection FiLM path after sweep evidence
        showed limited benefit from additional FiLM branches.

        Args:
            global_feat (Tensor["B, Nq, F_g"]): Global features to modulate.
            proj_feat (Tensor["B, Nq, F_proj"]): Projection summary features.
            film (nn.Module): MLP that outputs 2*F_g parameters (gamma, beta).
            norm (nn.GroupNorm | None): Optional normalization on channels.

        Returns:
            Tensor["B, Nq, F_g"]: FiLM-modulated global features.
        """
        film_out = film(proj_feat.to(dtype=global_feat.dtype))
        gamma, beta = film_out.chunk(2, dim=-1)
        global_feat = global_feat * (1.0 + gamma) + beta
        if norm is not None:
            global_feat = norm(global_feat.transpose(1, 2)).transpose(1, 2)
        return global_feat

    def _sample_semidense_points(
        self,
        snippet: VinSnippetView,
        *,
        device: torch.device,
    ) -> Tensor | None:
        """Sample semidense points once for shared use.

        Semidense projection stats were the most reliable signal in the sweep,
        so this enforces the (x, y, z, 1/sigma_d, n_obs) channel contract and
        fails fast on missing/invalid data.

        Args:
            snippet (VinSnippetView): Padded semidense snippet.
                - points_world (Tensor["B, P, C_sem"] or Tensor["P, C_sem"]):
                  channels (x, y, z, inv_dist_std, obs_count).
                - lengths (Tensor["B"], optional): valid lengths per batch item.
            device (torch.device): Target device for returned points.

        Returns:
            Tensor["B, P_fr, C_sem"] or Tensor["P_fr, C_sem"]:
                Subsampled semidense points with P_fr <= semidense_proj_max_points.
        """
        max_points = self.config.semidense_proj_max_points
        points = snippet.points_world
        lengths = getattr(snippet, "lengths", None)
        if points.numel() == 0:
            raise RuntimeError("VinSnippetView.points_world is empty.")
        if points.shape[-1] < 5:
            raise ValueError(
                "VinSnippetView.points_world must have at least 5 channels (x,y,z,1/sigma_d,n_obs).",
            )
        if points.ndim == 2:
            valid_len = None
            if lengths is not None and lengths.numel() > 0:
                valid_len = int(lengths.reshape(-1)[0].item())
                valid_len = min(valid_len, int(points.shape[0]))
            if valid_len is None:
                valid_len = int(points.shape[0])
            if valid_len <= 0:
                raise RuntimeError("VinSnippetView.points_world has zero valid points.")
            if valid_len > max_points:
                idx = torch.randperm(valid_len, device=points.device)[:max_points]
                points = points[:valid_len][idx]
            else:
                points = points[:valid_len]
            if not torch.isfinite(points[..., :3]).all():
                raise ValueError("VinSnippetView.points_world contains non-finite XYZ values.")
        elif points.ndim == 3:
            batch_size, _, dim = points.shape
            points_out = torch.full(
                (batch_size, max_points, dim),
                float("nan"),
                dtype=points.dtype,
                device=points.device,
            )
            for b in range(batch_size):
                if lengths is not None and lengths.numel() > b:
                    valid_len = int(lengths.reshape(-1)[b].item())
                    valid_len = min(valid_len, int(points.shape[1]))
                else:
                    valid_len = int(points.shape[1])
                if valid_len <= 0:
                    continue
                if valid_len > max_points:
                    idx = torch.randperm(valid_len, device=points.device)[:max_points]
                    points_out[b, :max_points] = points[b, :valid_len][idx]
                else:
                    points_out[b, :valid_len] = points[b, :valid_len]
            points = points_out
            if not torch.isfinite(points[..., :3]).all():
                raise ValueError("VinSnippetView.points_world contains non-finite XYZ values.")
        else:
            raise ValueError(
                f"Expected VinSnippetView.points_world with ndim 2 or 3, got {points.ndim}.",
            )
        return points.to(device=device, dtype=torch.float32)

    def _project_semidense_points(
        self,
        points_world: Tensor | None,
        p3d_cameras: PerspectiveCameras,
        *,
        batch_size: int,
        num_candidates: int,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Project points into candidate cameras and return screen coords + masks.

        This is the shared projection path for semidense stats and voxel FiLM,
        with strict camera batch checks to prevent silent misalignment.

        Args:
            points_world (Tensor["B, P, C_sem"] or Tensor["P, C_sem"]):
                Semidense points in world frame (XYZ + extras).
            p3d_cameras (PerspectiveCameras): Camera batch with size B*Nq.
            batch_size (int): B (batch size).
            num_candidates (int): Nq (candidates per batch).
            device (torch.device): Target device for projections.

        Returns:
            dict[str, Tensor]:
                x (Tensor["B*Nq, P_proj"]): Screen x (pixels).
                y (Tensor["B*Nq, P_proj"]): Screen y (pixels).
                z (Tensor["B*Nq, P_proj"]): Screen depth (positive in front).
                finite (Tensor["B*Nq, P_proj"]): Finite projection mask.
                valid (Tensor["B*Nq, P_proj"]): In-bounds + positive depth mask.
                inv_dist_std (Tensor["B*Nq, P_proj"] or empty): Optional per-point uncertainty.
                obs_count (Tensor["B*Nq, P_proj"] or empty): Optional per-point track length.
                image_size (Tensor["B*Nq, 2"]): (H, W) per camera.
                num_cams (Tensor[""]): Scalar camera count (B*Nq).
            Note:
                P_proj equals the number of points per camera after tiling:
                semidense points use P_proj = P_fr, voxel centers use P_proj = G_pool^3.
        """
        if points_world is None or points_world.numel() == 0:
            raise RuntimeError("Semidense projection requires non-empty points_world.")

        cameras = p3d_cameras.to(device)
        image_size = getattr(cameras, "image_size", None)
        if image_size is None or image_size.numel() == 0:
            raise RuntimeError("p3d_cameras.image_size is required for semidense projection.")

        num_cams = int(cameras.R.shape[0])
        if num_cams == 0:
            raise RuntimeError("p3d_cameras has zero cameras.")

        image_size = image_size.to(device=device, dtype=torch.float32)
        if image_size.shape[0] == 1 and num_cams > 1:
            image_size = image_size.expand(num_cams, -1)
        if image_size.shape[0] != num_cams:
            raise ValueError(
                f"p3d_cameras.image_size batch mismatch: image_size {tuple(image_size.shape)} vs num_cams={num_cams}.",
            )

        pts_world = points_world.to(device=device, dtype=torch.float32)
        xyz = pts_world[..., :3]
        extra = pts_world[..., 3:] if pts_world.shape[-1] > 3 else None
        inv_dist_std = None
        obs_count = None
        if extra is not None:
            if extra.shape[-1] >= 1:
                inv_dist_std = extra[..., 0]
            if extra.shape[-1] >= 2:
                obs_count = extra[..., 1]
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
            if inv_dist_std is not None:
                inv_dist_std = inv_dist_std.unsqueeze(0)
            if obs_count is not None:
                obs_count = obs_count.unsqueeze(0)
        if xyz.shape[0] == 1 and batch_size > 1:
            xyz = xyz.expand(batch_size, -1, -1)
            if inv_dist_std is not None:
                inv_dist_std = inv_dist_std.expand(batch_size, -1)
            if obs_count is not None:
                obs_count = obs_count.expand(batch_size, -1)
        if xyz.shape[0] != batch_size:
            raise ValueError("Semidense points batch size must match candidates.")

        if batch_size == 1 and num_cams == num_candidates:
            points_cam = xyz.expand(num_candidates, -1, -1)
            inv_cam = inv_dist_std.expand(num_candidates, -1) if inv_dist_std is not None else None
            obs_cam = obs_count.expand(num_candidates, -1) if obs_count is not None else None
        elif num_cams == batch_size * num_candidates:
            points_cam = xyz[:, None].expand(batch_size, num_candidates, -1, -1).reshape(num_cams, -1, 3)
            if inv_dist_std is not None:
                inv_cam = inv_dist_std[:, None].expand(batch_size, num_candidates, -1).reshape(num_cams, -1)
            else:
                inv_cam = None
            if obs_count is not None:
                obs_cam = obs_count[:, None].expand(batch_size, num_candidates, -1).reshape(num_cams, -1)
            else:
                obs_cam = None
        else:
            raise ValueError(
                "p3d_cameras batch size must be N (when B=1) or B*N; "
                f"got {num_cams} for B={batch_size}, N={num_candidates}.",
            )

        pts_screen = cameras.transform_points_screen(points_cam)
        x, y, z = pts_screen.unbind(dim=-1)
        h = image_size[:, 0].unsqueeze(1)
        w = image_size[:, 1].unsqueeze(1)
        finite = torch.isfinite(pts_screen).all(dim=-1)
        valid = finite & (z > 0.0) & (x >= 0.0) & (y >= 0.0) & (x <= (w - 1.0)) & (y <= (h - 1.0))

        return {
            "x": x,
            "y": y,
            "z": z,
            "finite": finite,
            "valid": valid,
            "inv_dist_std": inv_cam if inv_cam is not None else torch.empty(0, device=device),
            "obs_count": obs_cam if obs_cam is not None else torch.empty(0, device=device),
            "image_size": image_size,
            "num_cams": torch.tensor(num_cams, device=device),
        }

    def _encode_semidense_projection_features(
        self,
        proj_data: dict[str, Tensor] | None,
        *,
        batch_size: int,
        num_candidates: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Summarize projection coverage, visibility, and depth statistics.

        Coverage/visibility features are a lightweight proxy for frustum
        attention, and reliability weights use n_obs and 1/sigma_d to attenuate
        noisy points, matching sweep evidence without adding heavy modules.

        Args:
            proj_data (dict[str, Tensor] | None): Projection outputs.
                - x, y, z (Tensor["B*Nq, P_proj"]): screen coords + depth.
                - valid, finite (Tensor["B*Nq, P_proj"]): projection masks.
                - inv_dist_std (Tensor["B*Nq, P_proj"] or empty): optional uncertainty.
                - obs_count (Tensor["B*Nq, P_proj"] or empty): optional track length.
                - image_size (Tensor["B*Nq, 2"]): (H, W) per camera.
                - num_cams (Tensor[""]): scalar camera count.
            batch_size (int): B (batch size).
            num_candidates (int): Nq (candidates per batch).
            device (torch.device): Target device for features.
            dtype (torch.dtype): Output dtype.

        Returns:
            Tensor["B, Nq, F_proj"]: Projection summary features in order:
                [coverage, empty_frac, semidense_candidate_vis_frac, depth_mean, depth_std].
        """
        # Shapes (batched): x/y/z/valid are (B*N_q, P_proj); outputs are (B, N_q, F_proj).
        proj_feat = torch.zeros(
            (batch_size, num_candidates, SEMIDENSE_PROJ_DIM),
            device=device,
            dtype=dtype,
        )
        if proj_data is None:
            raise RuntimeError("Semidense projection data is missing.")

        x = proj_data["x"]  # (B*N_q, P_proj)
        y = proj_data["y"]  # (B*N_q, P_proj)
        z = proj_data["z"]  # (B*N_q, P_proj)
        finite = proj_data.get("finite")
        if finite is None:
            finite = torch.isfinite(torch.stack([x, y, z], dim=-1)).all(dim=-1)
        valid = proj_data["valid"]  # (B*N_q, P_proj)
        image_size = proj_data["image_size"]  # (B*N_q, 2) as (H, W)
        inv_dist_std = proj_data.get("inv_dist_std")  # (B*N_q, P_proj) or empty
        obs_count = proj_data.get("obs_count")  # (B*N_q, P_proj) or empty
        if inv_dist_std is not None and inv_dist_std.numel() == 0:
            inv_dist_std = None
        if obs_count is not None and obs_count.numel() == 0:
            obs_count = None
        num_cams = int(proj_data["num_cams"].item())
        h = image_size[:, 0].unsqueeze(1).clamp_min(1.0)
        w = image_size[:, 1].unsqueeze(1).clamp_min(1.0)

        # ------------------------------------------------------------------
        # 1) Coverage proxy: bin projected points onto a GxG grid (screen space).
        grid_size = self.config.semidense_proj_grid_size
        num_bins = grid_size * grid_size
        x_safe = torch.where(valid, x, torch.zeros_like(x))
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        z_safe = torch.where(valid, z, torch.zeros_like(z))
        x_safe = torch.nan_to_num(x_safe, nan=0.0, posinf=0.0, neginf=0.0)
        y_safe = torch.nan_to_num(y_safe, nan=0.0, posinf=0.0, neginf=0.0)
        z_safe = torch.nan_to_num(z_safe, nan=0.0, posinf=0.0, neginf=0.0)
        x_bin = torch.clamp((x_safe / w) * grid_size, 0.0, float(grid_size - 1)).to(dtype=torch.long)
        y_bin = torch.clamp((y_safe / h) * grid_size, 0.0, float(grid_size - 1)).to(dtype=torch.long)
        bin_idx = y_bin * grid_size + x_bin

        counts = torch.zeros((num_cams, num_bins), device=device, dtype=torch.float32)
        bin_idx = torch.where(valid, bin_idx, torch.zeros_like(bin_idx))
        valid_f = valid.to(dtype=counts.dtype)
        counts.scatter_add_(1, bin_idx, valid_f)
        coverage = (counts > 0).to(dtype=counts.dtype).mean(dim=1)
        empty_frac = 1.0 - coverage

        finite_f = finite.to(dtype=counts.dtype)
        eps = 1e-6
        # ------------------------------------------------------------------
        # 2) Reliability weights: combine n_obs (track length) and inv_dist_std.
        if obs_count is not None:
            obs = obs_count.to(device=device, dtype=counts.dtype).clamp_min(0.0)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs_log = torch.log1p(obs)
            obs_log = torch.where(finite, obs_log, torch.zeros_like(obs_log))
            denom = torch.log1p(
                torch.tensor(float(self.config.semidense_obs_count_max), device=device, dtype=counts.dtype)
            ).clamp_min(eps)
            a = (obs_log / denom).clamp(0.0, 1.0)
        else:
            # If n_obs is unavailable, treat all points as equally reliable.
            a = torch.ones_like(valid_f)

        if inv_dist_std is not None:
            inv = inv_dist_std.to(device=device, dtype=counts.dtype).clamp_min(0.0)
            inv = torch.nan_to_num(
                inv,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            inv_min = float(self.config.semidense_inv_dist_std_min)
            inv_p95 = max(float(self.config.semidense_inv_dist_std_p95), inv_min + eps)
            denom = torch.tensor(inv_p95 - inv_min, device=device, dtype=counts.dtype).clamp_min(eps)
            b = ((inv - inv_min) / denom).clamp(0.0, 1.0)
        else:
            # If 1/sigma_d is unavailable, treat all points as equally reliable.
            b = torch.ones_like(valid_f)
        w_rel = (a * b).clamp(0.0, 1.0)

        # ------------------------------------------------------------------
        # 3) Visibility proxy: weighted valid fraction among finite projections.
        weight_valid = w_rel * valid_f
        weight_finite = w_rel * finite_f
        weight_sum = weight_valid.sum(dim=1).clamp_min(eps)
        finite_sum = weight_finite.sum(dim=1).clamp_min(eps)
        semidense_candidate_vis_frac = weight_valid.sum(dim=1) / finite_sum

        # ------------------------------------------------------------------
        # 4) Depth stats: weighted mean/std of z over valid points.
        depth_mean = (z_safe * weight_valid).sum(dim=1) / weight_sum
        depth_var = ((z_safe - depth_mean.unsqueeze(1)) ** 2 * weight_valid).sum(dim=1) / weight_sum
        depth_std = torch.sqrt(depth_var.clamp_min(0.0))

        feats = torch.stack(
            [coverage, empty_frac, semidense_candidate_vis_frac, depth_mean, depth_std],
            dim=-1,
        )
        if batch_size == 1 and num_cams == num_candidates:
            proj_feat = feats.view(1, num_candidates, -1)
        else:
            proj_feat = feats.view(batch_size, num_candidates, -1)
        return proj_feat.to(device=device, dtype=dtype)

    @staticmethod
    def _semidense_proj_feature_index(name: str) -> int:
        if name in SEMIDENSE_PROJ_FEATURES:
            return SEMIDENSE_PROJ_FEATURES.index(name)
        alias = SEMIDENSE_PROJ_FEATURE_ALIASES.get(name)
        if alias is not None and alias in SEMIDENSE_PROJ_FEATURES:
            return SEMIDENSE_PROJ_FEATURES.index(alias)
        raise ValueError(f"Unknown semidense projection feature '{name}'.")

    def _forward_impl(
        self,
        efm: EfmSnippetView | VinSnippetView,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        return_debug: bool,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinV3ForwardDiagnostics | None]:
        """Run the VIN v3 forward pass.

        The pipeline enforces semidense availability, constructs the compact
        voxel field, applies pose-conditioned pooling and voxel FiLM, and uses
        CORAL ordinal regression to score candidates.
        # TODO: include a graph-like overview of the data flow within the _forward_impl in this doc-string. functions / modules / layers should be nodes; include shape and data information in the edges. i.e.

        Args:
            efm (EfmSnippetView | VinSnippetView): EFM or VIN snippet view.
            candidate_poses_world_cam (PoseTW["B, Nq, 12"]): Candidate camera poses T_w^c.
            reference_pose_world_rig (PoseTW["B, 12"]): Reference rig pose T_w^r.
            p3d_cameras (PerspectiveCameras): Camera batch size B*Nq (R, T, K).
            return_debug (bool): If True, return VinV3ForwardDiagnostics.
            backbone_out (EvlBackboneOutput | None): Optional cached EVL outputs.

        Returns:
            Tuple[VinPrediction, VinV3ForwardDiagnostics | None]:
                VinPrediction:
                    logits (Tensor["B, Nq, K-1"]): CORAL logits.
                    prob (Tensor["B, Nq, K"]): Ordinal probabilities.
                    expected (Tensor["B, Nq"]): Expected RRI.
                    expected_normalized (Tensor["B, Nq"]): Expected RRI in [0,1].
                    candidate_valid (Tensor["B, Nq"]): Valid candidate mask.
                    voxel_valid_frac (Tensor["B, Nq"]): Voxel coverage proxy.
                    semidense_candidate_vis_frac (Tensor["B, Nq"]): Semidense visibility proxy.
                VinV3ForwardDiagnostics (optional):
                    field_in (Tensor["B, F_in, D, H, W"]), field (Tensor["B, F_g, D, H, W"]),
                    global_feat (Tensor["B, Nq, F_g"]), semidense_proj (Tensor["B, Nq, F_proj"]),
                    voxel_proj (Tensor["B, Nq, F_proj"]), pos_grid (Tensor["B, 3, D, H, W"]), feats (Tensor["B, Nq, F_head"]).
        """
        # Shape notation (see docs/typst/shared/macros.typ):
        # B=batch size, N_q=#candidates (alias of N), D/H/W=voxel grid, V=voxel points,
        # P=points per snippet, P_proj=points per projection, F_*=feature dims.
        if self.config.apply_cw90_correction and not getattr(p3d_cameras, "cw90_corrected", False):
            raise RuntimeError(
                "apply_cw90_correction=True requires p3d_cameras to already be CW90-corrected. "
                "Set p3d_cameras.cw90_corrected = True after correcting the camera extrinsics, "
                "or disable apply_cw90_correction.",
            )
        # Inputs: candidate_poses_world_cam is PoseTW (B, N_q, 12),
        # reference_pose_world_rig is PoseTW (B, 12), p3d_cameras batch is B*N_q.
        efm_dict: dict[str, Any] | None
        if isinstance(efm, EfmSnippetView):
            efm_dict = efm.efm
        elif isinstance(efm, VinSnippetView):
            efm_dict = None
        else:
            raise TypeError(
                f"VinModelV3 expects a VinSnippetView or EfmSnippetView for `efm`, got {type(efm)}.",
            )
        if backbone_out is None:
            if self.backbone is None:  # type: ignore
                self.backbone = self.config.backbone.setup_target() if self.config.backbone is not None else None  # type: ignore
            if efm_dict is None:
                raise RuntimeError(
                    "VinModelV3 requires cached backbone outputs when using VinSnippetView.",
                )
            backbone_out = self.backbone.forward(efm_dict)  # type: ignore

        device = backbone_out.voxel_extent.device
        try:
            param_device = next(self.parameters()).device
        except StopIteration:
            param_device = device
        if param_device != device:
            self.to(device)

        vin_snippet = self._ensure_vin_snippet(efm, device=device)  # points_world: (B, P, C_sem)
        prepared = self._prepare_inputs(
            vin_snippet,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            backbone_out=backbone_out,
        )
        # prepared.pose_world_cam: (B, N_q, 12); pose_world_rig_ref: (B, 12)
        pose_feats = self._encode_pose_features(
            prepared.pose_world_cam,
            prepared.pose_world_rig_ref,
        )
        # pose_vec: (B, N_q, 9); pose_enc: (B, N_q, F_pose); candidate_center_rig_m: (B, N_q, 3)
        field_bundle = self._build_field_bundle(backbone_out)
        # field_in: (B, F_in, D, H, W); field: (B, F_g, D, H, W)

        candidate_centers_world = prepared.pose_world_cam.t.to(
            dtype=field_bundle.field.dtype,
        )  # (B, N_q, 3)
        counts_norm = field_bundle.aux.get("counts_norm")
        if counts_norm is None:
            raise KeyError("Missing counts_norm in field bundle.")
        center_tokens, center_valid = sample_voxel_field(
            counts_norm,
            points_world=candidate_centers_world.unsqueeze(2),
            t_world_voxel=prepared.t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        # center_tokens: (B, N_q, 1, 1); center_valid: (B, N_q, 1)
        center_valid = center_valid.squeeze(-1)  # (B, N_q)
        counts_norm_center = center_tokens[..., 0, 0]  # (B, N_q)
        pose_finite = torch.isfinite(pose_feats.pose_vec).all(dim=-1)
        voxel_valid_frac = (counts_norm_center * center_valid.to(dtype=counts_norm_center.dtype)).clamp(0.0, 1.0)
        voxel_valid_frac = (voxel_valid_frac * pose_finite.to(dtype=voxel_valid_frac.dtype)).clamp(0.0, 1.0)

        pts_world = backbone_out.pts_world
        if not isinstance(pts_world, torch.Tensor):
            raise KeyError(
                "Missing backbone output 'voxel/pts_world' required for positional encoding.",
            )
        global_ctx = self._compute_global_context(
            field_bundle.field,
            pose_feats.pose_enc,
            pts_world=pts_world,
            t_world_voxel=prepared.t_world_voxel,
            pose_world_rig_ref=prepared.pose_world_rig_ref,
            voxel_extent=backbone_out.voxel_extent,
        )
        # global_feat: (B, N_q, F_g); pos_grid: (B, 3, D, H, W)
        global_feat = global_ctx.global_feat
        if self.voxel_gate is not None:
            # Gate down candidates whose centers fall outside observed voxels.
            gate = self.voxel_gate(voxel_valid_frac.unsqueeze(-1).to(dtype=global_feat.dtype))
            global_feat = global_feat * gate
        pool_grid = min(
            int(self.config.global_pool_grid_size),
            int(field_bundle.field.shape[-3]),
            int(field_bundle.field.shape[-2]),
            int(field_bundle.field.shape[-1]),
        )
        voxel_points = self._pool_voxel_points(
            pts_world,
            grid_shape=(
                int(field_bundle.field.shape[-3]),
                int(field_bundle.field.shape[-2]),
                int(field_bundle.field.shape[-1]),
            ),
            pool_grid=pool_grid,
        )
        # voxel_points: (B, P_proj, 3) with P_proj = G_pool^3
        voxel_proj_data = self._project_semidense_points(
            voxel_points,
            p3d_cameras,
            batch_size=prepared.batch_size,
            num_candidates=prepared.num_candidates,
            device=prepared.device,
        )
        # voxel_proj_data: x/y/z/valid are (B*N_q, P_proj)
        voxel_proj = self._encode_semidense_projection_features(
            voxel_proj_data,
            batch_size=prepared.batch_size,
            num_candidates=prepared.num_candidates,
            device=prepared.device,
            dtype=field_bundle.field.dtype,
        )
        # voxel_proj: (B, N_q, F_proj=5)
        if self.voxel_proj_film is not None:
            global_feat = self._apply_film(
                global_feat,
                voxel_proj,
                film=self.voxel_proj_film,
                norm=self.voxel_proj_film_norm,
            )
        global_ctx = GlobalContext(pos_grid=global_ctx.pos_grid, global_feat=global_feat)
        semidense_points = self._sample_semidense_points(
            vin_snippet,
            device=prepared.device,
        )
        # semidense_points: (B, P_fr, C_sem)
        proj_data = self._project_semidense_points(
            semidense_points,
            p3d_cameras,
            batch_size=prepared.batch_size,
            num_candidates=prepared.num_candidates,
            device=prepared.device,
        )
        # proj_data: x/y/z/valid are (B*N_q, P_fr)
        semidense_proj = self._encode_semidense_projection_features(
            proj_data,
            batch_size=prepared.batch_size,
            num_candidates=prepared.num_candidates,
            device=prepared.device,
            dtype=field_bundle.field.dtype,
        )
        # semidense_proj: (B, N_q, F_proj=5)
        global_ctx = GlobalContext(pos_grid=global_ctx.pos_grid, global_feat=global_ctx.global_feat)

        semidense_idx = self._semidense_proj_feature_index("semidense_candidate_vis_frac")
        semidense_candidate_vis_frac = semidense_proj[..., semidense_idx]
        # candidate_valid: (B, N_q); require finite pose + observed voxel + visible semidense.
        candidate_valid = pose_finite & (voxel_valid_frac > 0.0) & (semidense_candidate_vis_frac > 0.0)

        # ------------------------------------------------------------------ final feature assembly + scoring
        parts: list[Tensor] = [
            pose_feats.pose_enc.to(device=prepared.device, dtype=field_bundle.field.dtype),
            global_feat,
        ]

        feats = torch.cat(parts, dim=-1)  # (B, N_q, F_head)
        flat_feats = feats.reshape(prepared.batch_size * prepared.num_candidates, -1)  # (B*N_q, F_head)
        logits = self.head_coral(self.head_mlp(flat_feats)).reshape(
            prepared.batch_size,
            prepared.num_candidates,
            -1,
        )
        # logits: (B, N_q, K-1)

        prob = coral_logits_to_prob(logits)
        expected, expected_norm = coral_expected_from_logits(logits)  # (B, N_q)

        pred = VinPrediction(
            logits=logits,
            prob=prob,
            expected=expected,
            expected_normalized=expected_norm,
            candidate_valid=candidate_valid,
            voxel_valid_frac=voxel_valid_frac,
            semidense_candidate_vis_frac=semidense_candidate_vis_frac,
            semidense_valid_frac=semidense_candidate_vis_frac,
        )

        if not return_debug:
            return pred, None

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------ diagnostics

        debug = VinV3ForwardDiagnostics(
            backbone_out=backbone_out,
            candidate_center_rig_m=pose_feats.candidate_center_rig_m,
            pose_enc=pose_feats.pose_enc,
            pose_vec=pose_feats.pose_vec,
            field_in=field_bundle.field_in,
            field=field_bundle.field,
            global_feat=global_feat,
            candidate_valid=candidate_valid,
            voxel_valid_frac=voxel_valid_frac,
            semidense_candidate_vis_frac=semidense_candidate_vis_frac,
            semidense_valid_frac=semidense_candidate_vis_frac,
            pos_grid=global_ctx.pos_grid,
            feats=feats,
            semidense_proj=semidense_proj,
            voxel_proj=voxel_proj,
        )
        return pred, debug

    def forward(
        self,
        efm: EfmSnippetView | VinSnippetView,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> VinPrediction:
        """Score candidate poses for one snippet (no diagnostics).

        Args:
            efm: EFM snippet view or VIN snippet view for the current snippet.
            candidate_poses_world_cam: Candidate camera poses in world frame.
            reference_pose_world_rig: Reference rig pose in world frame.
            p3d_cameras: Pytorch3D cameras aligned with candidates.
            backbone_out: Optional precomputed backbone output.

        Returns:
            VinPrediction containing ordinal logits, expected scores, and
            validity masks for each candidate.
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
        efm: EfmSnippetView | VinSnippetView,
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinV3ForwardDiagnostics]:
        """Run VIN v3 forward pass and return intermediate tensors.

        Args:
            efm: EFM snippet view or VIN snippet view for the current snippet.
            candidate_poses_world_cam: Candidate camera poses in world frame.
            reference_pose_world_rig: Reference rig pose in world frame.
            p3d_cameras: Pytorch3D cameras aligned with candidates.
            backbone_out: Optional precomputed backbone output.

        Returns:
            Tuple of (VinPrediction, VinV3ForwardDiagnostics).
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
            raise RuntimeError(
                "Expected VinV3ForwardDiagnostics when return_debug=True.",
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
        """Summarize VIN v3 inputs/outputs for a single oracle-labeled batch.

        This is intended for quick sanity checks when validating the
        streamlined baseline against sweep-derived expectations.

        Args:
            batch: Oracle-labeled batch to inspect.
            include_torchsummary: Whether to include a torchsummary block.
            torchsummary_depth: Depth for torchsummary.

        Returns:
            Human-readable summary string.
        """
        return summarize_vin_v3(
            self, batch, include_torchsummary=include_torchsummary, torchsummary_depth=torchsummary_depth
        )

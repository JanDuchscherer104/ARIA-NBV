"""VIN (View Introspection Network) head on top of a frozen EVL backbone.

This module implements the *learned* component of NBV scoring that sits on top of
EVL's frozen voxel-lifting features. The goal is to predict the **Relative
Reconstruction Improvement (RRI)** for each candidate view, which the oracle
defines (conceptually) as:

    RRI = (d_before - d_after) / d_before,

where ``d_before`` and ``d_after`` are point-to-mesh distances (e.g., Chamfer-like
metrics) before and after adding a candidate view. In training, we discretize
RRI into ordinal bins and optimize the CORAL ordinal loss; at inference we use
the expected ordinal value as a normalized score.

Coordinate frames and transforms follow the EFM3D/ATEK conventions:

- ``T_A_B`` is a transform mapping points from frame **B** to frame **A**.
- ``PoseTW`` stores SE(3) as ``(R, t)`` in world units.
- ``world`` is the global frame; ``rig`` is the device frame; ``cam`` is a
  specific camera frame; ``voxel`` is EVL's voxel grid frame.

Key ingredients implemented here:

1. **Learnable Fourier pose encoding** (candidate pose in reference frame)

   Given the relative pose:

       T_rig_ref_cam = T_world_rig_ref^{-1} * T_world_cam,

   we define:

       t = translation(T_rig_ref_cam)        (candidate center in rig-ref),
       r = ||t||                               (radius),
       u = t / (r + eps)                       (center direction),
       f = R_rig_ref_cam * z_cam               (camera forward direction),
       s = <f, -u>                             (view alignment scalar).

   These are concatenated into a single pose vector

       x = [u, f, r, s] ∈ R^8,

   and encoded by ``LearnableFourierFeatures`` (LFF). Alternatively, VIN can
   use a simplified translation + 6D rotation encoding:

       x = [t, R_{6d}] ∈ R^9,

   with learned per-group scaling for translation vs rotation.

2. **Scene field construction**

   EVL provides voxel-aligned evidence volumes. We build a compact field
   ``F(v) in R^{C_in}`` with channels such as:

       occ_pr          = P(occupied | EVL), in [0,1]
       occ_input       = voxelized occupancy evidence from input points
       counts_norm     = log1p(counts) / log1p(max_counts)
       observed        = 1[counts > 0]
       unknown         = 1 - observed
       new_surface_prior = unknown * occ_pr
       free_input      = EVL free-space evidence (or proxy)

   The field is projected with ``1x1x1 Conv3d + GroupNorm + GELU`` to a learned
   feature dimension.

3. **Candidate-conditioned voxel query (frustum sampling)**

   For each candidate camera we sample a small grid of rays on the image plane
   and a set of metric depths ``{d_i}``. The resulting points are unprojected
   into world coordinates using PyTorch3D and then mapped into the voxel frame:

       p_voxel = T_voxel_world * p_world,
       i_x = (x - x_min) / dx,  dx = (x_max - x_min) / W,
       i_y = (y - y_min) / dy,  dy = (y_max - y_min) / H,
       i_z = (z - z_min) / dz,  dz = (z_max - z_min) / D.

   We then sample ``F`` with trilinear interpolation (``grid_sample``), yielding
   per-ray tokens and a validity mask.

4. **Global tokens + CORAL head**

   The per-candidate features are:

       [pose_enc, global_feat?, voxel_pose_enc?, local_frustum_feat, voxel_valid_frac?],

   where ``global_feat`` can be pose-conditioned via attention pooling, and
   ``voxel_valid_frac`` summarizes frustum coverage (low coverage can still signal
   high RRI due to unknown space).

   The head outputs CORAL logits ``l_k`` for thresholds ``k=0..K-2``:

       P(y > k) = sigmoid(l_k),
       E[y] = sum_k P(y > k),
       E_norm = E[y] / (K-1).

These steps provide an interpretable mapping from candidate pose + EVL scene
context to an ordinal NBV score that correlates with oracle RRI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field, field_validator, model_validator
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)
from pytorch3d.transforms import matrix_to_rotation_6d  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.nn import functional as functional

from ...data.efm_views import EfmSnippetView, VinSnippetView
from ...rri_metrics.coral import (
    CoralLayer,
    coral_expected_from_logits,
    coral_logits_to_prob,
)
from ...utils import BaseConfig
from ..backbone_evl import EvlBackboneConfig
from .pose_encoding import LearnableFourierFeaturesConfig
from .types import EvlBackboneOutput, VinForwardDiagnostics, VinPrediction

if TYPE_CHECKING:
    from ...data.vin_oracle_types import VinOracleBatch


def _largest_divisor_leq(n: int, max_divisor: int) -> int:
    """Return the largest divisor of ``n`` that is <= ``max_divisor``.

    This helper is used to choose a valid GroupNorm group count. GroupNorm
    requires ``num_groups`` to divide ``num_channels`` exactly. We therefore
    compute:

        g = max { d : d <= max_divisor and n % d == 0 }.

    Args:
        n: Channel dimension to be normalized.
        max_divisor: Upper bound for the group count.

    Returns:
        Largest valid group count (>=1).
    """
    g = min(max_divisor, n)
    while g > 1 and (n % g) != 0:
        g -= 1
    return max(1, g)


def _build_frustum_points_world_p3d(
    cameras: PerspectiveCameras,
    *,
    grid_size: int,
    depths_m: list[float],
) -> Tensor:
    """Unproject a small frustum grid into world points at fixed metric depths.

    Conceptually, we build a sparse set of rays in the image plane and sample
    points along each ray at a list of metric depths ``{d_i}``. The point set is
    intentionally aligned with the PyTorch3D depth renderer so that:

        - a rendered depth map at depth ``d`` corresponds to the same 3D ray
          geometry used for VIN's voxel query, and
        - the derived point clouds used in the oracle pipeline are consistent
          with VIN's local feature sampling.

    The procedure is:

    1) Build a symmetric ``grid_size x grid_size`` pixel grid around the
       principal point, clamped to valid pixel *centers*:

           u in [0.5, W - 0.5],  v in [0.5, H - 0.5].

    2) Convert to NDC coordinates (PyTorch3D uses +X left, +Y up):

           x_ndc = -(u - 0.5 * W) * (2 / scale),
           y_ndc = -(v - 0.5 * H) * (2 / scale),

       where ``scale = min(H, W)`` ensures square normalization when the image
       is not square.

    3) Stack with metric depths ``z = d_i`` and call
       ``cameras.unproject_points(..., from_ndc=True)`` to obtain world points.

    Args:
        cameras: PyTorch3D ``PerspectiveCameras`` (screen-space intrinsics).
        grid_size: Number of samples per image axis (total rays = grid_size^2).
        depths_m: List of metric depths in meters along each ray.

    Returns:
        ``Tensor["B (grid_size^2 * len(depths_m)) 3", float32]`` world points
        for each camera in the batch.
    """
    num_cams = int(cameras.R.shape[0])
    device = cameras.R.device

    # Screen-space camera inputs are in pixels, but `unproject_points(..., from_ndc=True)`
    # expects NDC coordinates (+X left, +Y up) where the conversion depends on `image_size`.
    image_size = cameras.image_size.to(device=device, dtype=torch.float32)
    principal_point = cameras.principal_point.to(device=device, dtype=torch.float32)
    if image_size.shape[0] == 1 and num_cams > 1:
        image_size = image_size.expand(num_cams, -1)
    if principal_point.shape[0] == 1 and num_cams > 1:
        principal_point = principal_point.expand(num_cams, -1)

    h = image_size[:, 0]
    w = image_size[:, 1]
    scale = torch.minimum(h, w)

    # Sample a small pixel grid around the principal point, then convert to NDC.
    #
    # Keep the sampling range symmetric around the principal point (important when the
    # principal point is not exactly at the image center) while clamping to valid pixel
    # *centers* in [0.5, W-0.5]×[0.5, H-0.5].
    half = 0.95 * 0.5 * scale
    half_x = torch.minimum(
        half,
        torch.minimum(principal_point[:, 0] - 0.5, (w - 0.5) - principal_point[:, 0]),
    )
    half_y = torch.minimum(
        half,
        torch.minimum(principal_point[:, 1] - 0.5, (h - 0.5) - principal_point[:, 1]),
    )
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
    """Build a compact voxel-aligned scene field from EVL head/evidence tensors.

    EVL exposes a set of voxel grids with different semantics. VIN collapses
    those into a small channel tensor ``F(v)`` that can be sampled at candidate
    frustum points. The supported channels are:

    - ``occ_pr``: occupancy probability (or logits if ``occ_pr_is_logits=True``):

          occ_pr = sigmoid(occ_pr_logits)  (if logits)

    - ``occ_input``: binary occupancy evidence from the input points.
    - ``counts_norm``: normalized observation counts:

          counts_norm = log1p(counts) / log1p(max(counts))   (log1p mode)
          counts_norm = counts / max(counts)                (linear mode)

    - ``observed``: 1[counts > 0]
    - ``unknown``: 1 - observed
    - ``new_surface_prior``: unknown * occ_pr
    - ``cent_pr``: EVL centerness probability (if provided by the backbone)
    - ``free_input``: explicit free-space evidence if available; otherwise a
      weak proxy:

          free_input ~= observed * (1 - occ_input > threshold)

    Args:
        out: Backbone output bundle (must include head/evidence tensors).
        use_channels: Ordered list of channel names to concatenate.
        occ_input_threshold: Threshold used when deriving fallback free-space evidence.
        counts_norm_mode: Normalization mode for counts ("log1p" or "linear").
        occ_pr_is_logits: Whether `occ_pr` are logits (apply sigmoid) rather than
            probabilities.

    Returns:
        ``Tensor["B C D H W", float32]`` scene field with channels in the
        same order as ``use_channels``.
    """

    def _require(name: str) -> Tensor:
        value = getattr(out, name)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Missing backbone output '{name}'. Ensure EvlBackboneConfig.features_mode includes 'heads'.",
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

    if "cent_pr" in use_channels:
        parts["cent_pr"] = _require("cent_pr").to(dtype=torch.float32)

    if "free_input" in use_channels:
        if isinstance(out.free_input, torch.Tensor):
            parts["free_input"] = out.free_input.to(dtype=torch.float32)
        else:
            # Fallback: derive a weak free-space proxy from (counts, occ_input).
            counts = _require("counts")
            observed = (counts > 0).to(dtype=torch.float32).unsqueeze(1)
            occ_evidence = (parts["occ_input"] > occ_input_threshold).to(
                dtype=torch.float32,
            )
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
            parts["counts_norm"] = torch.log1p(counts).unsqueeze(1) / torch.log1p(
                max_counts,
            ).unsqueeze(1)
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

    We map world-space points into EVL's voxel frame using the provided
    ``voxel/T_world_voxel`` pose:

        T_voxel_world = (T_world_voxel)^{-1}
        p_voxel = T_voxel_world * p_world.

    The voxel frame is **metric** (meters). We convert metric coordinates to
    voxel indices via the extent bounds:

        i_x = (x - x_min) / dx,  dx = (x_max - x_min) / W,
        i_y = (y - y_min) / dy,  dy = (y_max - y_min) / H,
        i_z = (z - z_min) / dz,  dz = (z_max - z_min) / D.

    ``pc_to_vox`` returns both these indices and an *extent* validity mask.
    ``sample_voxels`` then performs trilinear interpolation in grid coordinates
    (``grid_sample`` under the hood) and returns a *grid* validity mask.

    Args:
        field: ``Tensor["B C D H W"]`` voxel-aligned feature field.
        points_world: ``Tensor["B N K 3"]`` world points (K points per candidate).
        t_world_voxel: ``PoseTW["B 12"]`` world<-voxel transform.
        voxel_extent: ``Tensor["B 6"]`` voxel grid extent in voxel frame
            ``[x_min,x_max,y_min,y_max,z_min,z_max]``.

    Returns:
        Tuple of:
            - tokens: ``Tensor["B N K C", float32]`` sampled features.
            - valid: ``Tensor["B N K", bool]`` mask of in-bounds samples
              (extent AND grid validity).
    """
    if field.ndim != 5:
        raise ValueError(f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.")
    if points_world.ndim != 4:
        raise ValueError(
            f"Expected points_world shape (B,N,K,3), got {tuple(points_world.shape)}.",
        )
    if int(points_world.shape[-1]) != 3:
        raise ValueError(
            f"Expected points_world[..., 3], got {tuple(points_world.shape)}.",
        )

    batch_size, field_channels, grid_d, grid_h, grid_w = field.shape
    _, num_candidates, num_points, _ = points_world.shape

    t_world_voxel_b = t_world_voxel
    if t_world_voxel_b.ndim == 1:
        t_world_voxel_b = PoseTW(t_world_voxel_b._data.unsqueeze(0))
    if int(t_world_voxel_b.shape[0]) != int(batch_size):
        if int(t_world_voxel_b.shape[0]) == 1:
            t_world_voxel_b = PoseTW(t_world_voxel_b._data.expand(batch_size, 12))
        else:
            raise ValueError(
                "t_world_voxel must have batch size 1 or match field batch size.",
            )

    vox_extent = voxel_extent.to(device=field.device, dtype=torch.float32)
    if vox_extent.ndim == 1:
        vox_extent = vox_extent.view(1, 6).expand(batch_size, 6)
    if vox_extent.shape != (batch_size, 6):
        raise ValueError(
            f"Expected voxel_extent shape (B,6), got {tuple(vox_extent.shape)}.",
        )

    world_points_flat = points_world.to(device=field.device, dtype=field.dtype).reshape(
        batch_size,
        num_candidates * num_points,
        3,
    )

    # NOTE: EVL's voxel field is defined in the *voxel frame* (metres), but our candidates/frustum points are in WORLD.
    # EVL provides `voxel/T_world_voxel` (world←voxel). We invert it to get voxel←world and map points into voxel coords.
    # NOTE: If you ever swap EVL conventions or change voxel-grid anchoring, re-verify this transform (sanity check:
    # voxelized points should be stable under small candidate translations).
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

    samp, valid_grid = sample_voxels(
        field,
        pts_vox_id,
        differentiable=False,
    )  # B C (N*K), B (N*K)
    valid = (valid_extent & valid_grid).reshape(batch_size, num_candidates, num_points)
    tokens = samp.transpose(1, 2).reshape(
        batch_size,
        num_candidates,
        num_points,
        field_channels,
    )
    return tokens, valid


def _candidate_valid_from_token(
    token_valid: Tensor,
    *,
    min_valid_frac: float,
) -> Tensor:
    """Convert per-token validity into a per-candidate mask.

    For each candidate we compute the fraction of in-bounds samples:

        valid_frac = (1 / K) * sum_k 1[valid_k],

    and keep the candidate if ``valid_frac >= min_valid_frac``. This prevents
    degenerate candidates whose frustum points mostly fall outside the voxel
    grid (e.g., due to camera pose mismatch or extreme viewpoints).

    Args:
        token_valid: ``Tensor["B N K", bool]`` validity per frustum sample.
        min_valid_frac: Minimum fraction of valid samples to accept a candidate.

    Returns:
        ``Tensor["B N", bool]`` candidate validity mask.
    """
    if token_valid.ndim < 1:
        raise ValueError(
            f"Expected token_valid with ndim>=1, got {tuple(token_valid.shape)}.",
        )
    valid_frac = token_valid.float().mean(dim=-1)
    return valid_frac >= min_valid_frac


class PoseConditionedGlobalPool(nn.Module):
    """Pose-conditioned attention pooling over a coarse voxel grid.

    This module downsamples the voxel field to a coarse grid, flattens it into
    tokens, and applies multi-head attention with the candidate pose embeddings
    as queries. The result is a global context token per candidate that
    preserves spatial structure while remaining lightweight.
    """

    def __init__(
        self,
        *,
        field_dim: int,
        pose_dim: int,
        pool_size: int,
        attn_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads}).",
            )
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0.")
        self.pool_size = int(pool_size)
        self.attn_dim = int(attn_dim)
        self.pool = nn.AdaptiveAvgPool3d(
            (self.pool_size, self.pool_size, self.pool_size),
        )
        self.kv_proj = nn.Linear(field_dim, self.attn_dim)
        self.q_proj = nn.Linear(pose_dim, self.attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, field: Tensor, pose_enc: Tensor) -> Tensor:
        """Return pose-conditioned global tokens.

        Args:
            field: ``Tensor["B C D H W", float32]`` voxel field.
            pose_enc: ``Tensor["B N E_pose", float32]`` pose embeddings.

        Returns:
            ``Tensor["B N E_attn", float32]`` global tokens.
        """
        if field.ndim != 5:
            raise ValueError(
                f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.",
            )
        if pose_enc.ndim != 3:
            raise ValueError(
                f"Expected pose_enc shape (B,N,E), got {tuple(pose_enc.shape)}.",
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
        queries = self.q_proj(pose_enc.to(dtype=keys.dtype))
        attn_out, _ = self.attn(queries, keys, keys, need_weights=False)
        return attn_out


class VinScorerHead(nn.Module):
    """Candidate scoring head producing CORAL ordinal logits.

    VIN follows VIN-NBV by framing RRI prediction as **ordinal regression**.
    The head maps per-candidate features to ``K-1`` threshold logits:

        logit_k = w^T h + b_k,  k = 0..K-2,

    which parameterize the probabilities:

        P(y > k) = sigmoid(logit_k).

    This structure preserves the ordering between bins and allows the CORAL
    loss to penalize mis-ranked predictions more gracefully than MSE on raw
    RRI values.
    """

    def __init__(
        self,
        config: VinScorerHeadConfig,
        *,
        in_dim: int | None = None,
    ) -> None:
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
        """Compute CORAL logits from per-candidate features.

        Args:
            x: ``Tensor["... F"]`` input features (flattened over batch/candidates).

        Returns:
            ``Tensor["... K-1"]`` CORAL threshold logits.
        """
        return self.coral(self.mlp(x))


class VinScorerHeadConfig(BaseConfig[VinScorerHead]):
    """Configuration for :class:`VinScorerHead`.

    The head is a shallow MLP followed by a CORAL layer. The MLP produces a
    shared latent ``h`` for all thresholds, while CORAL adds independent biases
    per threshold. This enforces monotonic ordering in the ordinal space and
    reduces parameter count compared to a full K-way classifier.
    """

    @property
    def target(self) -> type[VinScorerHead]:
        """Factory target for :meth:`BaseConfig.setup_target`."""
        return VinScorerHead

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
        return self.target(self, in_dim=in_dim)


class VinModelConfig(BaseConfig["VinModel"]):
    """Configuration for :class:`VinModel`.

    This config collects all architectural choices that determine how VIN
    represents scene context and candidate poses. Conceptually, the VIN score is
    a function:

        score = f( pose_enc(u,f,r,s),
                   voxel_pose_enc?,
                   global_feat?,
                   local_frustum_feat,
                   voxel_valid_frac? ),

    where ``pose_enc`` and ``voxel_pose_enc`` are LFF-based shell encodings,
    ``global_feat`` summarizes the voxel field (optionally pose-conditioned),
    and ``local_frustum_feat`` samples the voxel field along a candidate frustum.
    """

    @property
    def target(self) -> type[VinModel]:
        """Factory target for :meth:`BaseConfig.setup_target`."""
        return VinModel

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Optional frozen EVL backbone configuration."""

    pose_encoder_lff: LearnableFourierFeaturesConfig = Field(
        default_factory=lambda: LearnableFourierFeaturesConfig(input_dim=9),
        validation_alias="pose_encoder_sh",
    )
    """Learnable Fourier Features pose encoding configuration (per pose vector)."""

    pose_encoding_mode: Literal["shell_lff", "t_r6d_lff"] = "t_r6d_lff"
    """Pose vector used for LFF: shell descriptor or translation + rotation-6D."""

    pose_scale_init: tuple[float, float] = (1.0, 1.0)
    """Initial per-group scale (translation, rotation-6D) applied before LFF."""

    pose_scale_learnable: bool = True
    """Whether pose scaling parameters are learned."""

    pose_scale_eps: float = Field(default=1e-6, gt=0.0)
    """Numerical floor for pose scaling (softplus + eps)."""

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
    ] = Field(
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
    """Whether to concatenate a global context token to per-candidate features."""

    global_pool_mode: Literal["mean", "mean_max", "attn"] = "attn"
    """Global pooling mode: mean, mean+max, or pose-conditioned attention."""

    global_pool_grid_size: int = Field(default=8, gt=0)
    """Target grid size for attention pooling (downsampled voxel resolution)."""

    global_pool_dim: int | None = None
    """Attention embedding dimension (defaults to `field_dim` when None)."""

    global_pool_heads: int = Field(default=4, gt=0)
    """Number of attention heads for pose-conditioned pooling."""

    global_pool_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Dropout rate for attention pooling."""

    use_voxel_pose_encoding: bool = True
    """Whether to append a LFF-encoded voxel-grid pose (voxel/T_world_voxel) in the reference frame."""

    use_unknown_token: bool = True
    """Whether to replace invalid frustum samples with a learned unknown token."""

    use_valid_frac_feature: bool = True
    """Whether to append (voxel_valid_frac, 1-voxel_valid_frac) as scalar features."""

    candidate_min_valid_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    """Minimum fraction of valid frustum samples required to keep a candidate."""

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

    @field_validator("pose_encoder_lff")
    @classmethod
    def _validate_pose_encoder_lff(
        cls,
        value: LearnableFourierFeaturesConfig,
    ) -> LearnableFourierFeaturesConfig:
        """Ensure the LFF input dimensionality matches the pose vector definition."""
        if value.input_dim not in (8, 9):
            raise ValueError(
                "pose_encoder_lff.input_dim must be 8 (shell) or 9 (t+R6d).",
            )
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
        if len(value) != 2:
            raise ValueError("pose_scale_init must be a (translation, rotation) tuple.")
        return value

    @model_validator(mode="after")
    def _validate_pose_encoding_mode(self) -> VinModelConfig:
        expected = 8 if self.pose_encoding_mode == "shell_lff" else 9
        if self.pose_encoder_lff.input_dim != expected:
            raise ValueError(
                "pose_encoder_lff.input_dim must match pose_encoding_mode "
                f"({self.pose_encoding_mode} expects {expected}).",
            )
        return self


class VinModel(nn.Module):
    """View Introspection Network (VIN) predicting RRI from EVL voxel features + pose.

    VIN is a light-weight head that queries frozen EVL voxel features to score
    candidate camera poses. The architecture is deliberately simple:

    - **Pose encoding** via learnable Fourier features over either the shell
      descriptor ``[u, f, r, s]`` or the simplified ``[t, R6d]`` vector.
    - **Scene field** built from EVL evidence volumes and projected with a
      1x1x1 Conv3d to a small feature dimension.
    - **Local query**: sample the scene field at frustum points and pool.
    - **Global tokens**: optional pose-conditioned attention pooling (or mean/mean+max)
      plus optional voxel-pose token.
    - **CORAL head** to produce ordinal scores.

    The overall score is computed as:

        z = concat(pose_enc, global_feat?, voxel_pose_enc?, local_feat, voxel_valid_frac?)
        logits = CORAL(MLP(z))
        score = E[y]/(K-1) = (1/(K-1)) * sum_k sigmoid(logit_k)
    """

    def __init__(self, config: VinModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = None
        self.pose_encoder_lff = self.config.pose_encoder_lff.setup_target()

        field_dim = self.config.field_dim
        gn_groups = _largest_divisor_leq(field_dim, self.config.field_gn_groups)

        field_in_dim = len(self.config.scene_field_channels)
        self.field_proj = nn.Sequential(
            nn.Conv3d(field_in_dim, field_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )

        self.use_global_pool = self.config.use_global_pool
        self.global_pool_mode = self.config.global_pool_mode
        self.use_unknown_token = self.config.use_unknown_token
        self.use_valid_frac_feature = self.config.use_valid_frac_feature
        self.use_voxel_pose_encoding = self.config.use_voxel_pose_encoding
        self.pose_encoding_mode = self.config.pose_encoding_mode
        self.pose_scale_eps = float(self.config.pose_scale_eps)

        # Head input dim is data-dependent (feature channel count depends on EVL cfg).
        pose_dim = int(self.pose_encoder_lff.out_dim)
        head_in_dim = pose_dim + field_dim

        self.global_pooler: PoseConditionedGlobalPool | None = None
        self.global_pool_dim = field_dim
        if self.use_global_pool:
            if self.global_pool_mode == "attn":
                attn_dim = self.config.global_pool_dim or field_dim
                if attn_dim % int(self.config.global_pool_heads) != 0:
                    raise ValueError(
                        "global_pool_dim must be divisible by global_pool_heads.",
                    )
                self.global_pool_dim = int(attn_dim)
                self.global_pooler = PoseConditionedGlobalPool(
                    field_dim=field_dim,
                    pose_dim=pose_dim,
                    pool_size=int(self.config.global_pool_grid_size),
                    attn_dim=int(attn_dim),
                    num_heads=int(self.config.global_pool_heads),
                    dropout=float(self.config.global_pool_dropout),
                )
            elif self.global_pool_mode == "mean_max":
                self.global_pool_dim = 2 * field_dim
            elif self.global_pool_mode == "mean":
                self.global_pool_dim = field_dim
            else:
                raise ValueError(f"Unknown global_pool_mode '{self.global_pool_mode}'.")
            head_in_dim += self.global_pool_dim
        if self.use_voxel_pose_encoding:
            head_in_dim += pose_dim
        if self.use_valid_frac_feature:
            head_in_dim += 2

        self.unknown_token: nn.Parameter | None = None
        if self.use_unknown_token:
            self.unknown_token = nn.Parameter(torch.zeros(1, 1, 1, field_dim))

        scale_init = torch.tensor(self.config.pose_scale_init, dtype=torch.float32)
        if self.config.pose_scale_learnable:
            self.pose_scale_log = nn.Parameter(torch.log(scale_init))
        else:
            self.register_buffer(
                "pose_scale_log",
                torch.log(scale_init),
                persistent=False,
            )
        self.head = self.config.head.setup_target(in_dim=head_in_dim)
        device = self.backbone.device if self.backbone is not None else torch.device("cpu")
        self.to(device)

    def _pose_scales(self) -> Tensor:
        """Return positive per-group scales for translation and rotation."""
        scales = functional.softplus(self.pose_scale_log) + self.pose_scale_eps
        return scales.to(dtype=torch.float32)

    def _pool_global(self, field: Tensor, pose_enc: Tensor) -> Tensor:
        """Pool a global context token from a voxel field.

        Modes:
            - ``mean``: global mean over the voxel grid.
            - ``mean_max``: concatenated mean + max.
            - ``attn``: pose-conditioned attention over a coarse voxel grid.

        Args:
            field: ``Tensor["B C D H W", float32]`` projected scene field.
            pose_enc: ``Tensor["B N E_pose", float32]`` pose embeddings.

        Returns:
            ``Tensor["B N C_global", float32]`` global tokens.
        """
        batch_size, num_candidates = int(pose_enc.shape[0]), int(pose_enc.shape[1])

        match self.global_pool_mode:
            case "mean":
                pooled = field.mean(dim=(-3, -2, -1))
                return pooled.unsqueeze(1).expand(batch_size, num_candidates, -1)
            case "mean_max":
                mean = field.mean(dim=(-3, -2, -1))
                maxv = field.amax(dim=(-3, -2, -1))
                pooled = torch.cat([mean, maxv], dim=-1)
                return pooled.unsqueeze(1).expand(batch_size, num_candidates, -1)
            case "attn":
                if self.global_pooler is None:
                    raise RuntimeError(
                        "global_pooler not initialized for attention pooling.",
                    )
                return self.global_pooler(field, pose_enc)
            case _:
                raise ValueError(f"Unknown global_pool_mode '{self.global_pool_mode}'.")

    def _frustum_points_world(
        self,
        poses_world_cam: PoseTW,
        *,
        p3d_cameras: PerspectiveCameras,
    ) -> Tensor:
        """Generate frustum sample points in world coordinates for each candidate.

        This is a thin wrapper around ``_build_frustum_points_world_p3d`` that
        reshapes the returned points into ``(B, N, K, 3)``. The function assumes
        that ``p3d_cameras`` is ordered to match the candidates:

        - If ``B=1``, then ``p3d_cameras`` must have batch size ``N``.
        - Otherwise it must have batch size ``B*N``.

        Returns:
            ``Tensor["B N K 3"]`` world points (K = grid_size^2 * len(depths_m)).
        """
        poses = poses_world_cam
        if poses.ndim != 3:
            raise ValueError(
                "poses_world_cam must have shape (B,N,12). Use `_ensure_candidate_batch` before calling this helper.",
            )
        batch_size = int(poses.t.shape[0])
        num_candidates = int(poses.t.shape[1])

        cameras = p3d_cameras.to(device=poses.t.device)
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

    @staticmethod
    def _pool_candidates(
        *,
        tokens: Tensor,
        valid: Tensor,
        unknown_token: Tensor | None = None,
    ) -> Tensor:
        """Pool candidate-local frustum samples.

        If ``unknown_token`` is provided, invalid samples are replaced with a
        learnable embedding and a simple mean over K is used:

            local_feat = mean_k token_k (invalid samples → unknown_token).

        Otherwise we compute a masked mean:

            local_feat = sum_k (valid_k * token_k) / (sum_k valid_k + eps).

        Args:
            tokens: ``Tensor["B N K C", float32]`` sampled features.
            valid: ``Tensor["B N K", bool]`` validity mask.
            unknown_token: Optional ``Tensor["1 1 1 C", float32]`` learnable token.

        Returns:
            ``Tensor["B N C", float32]`` pooled local features.
        """
        if tokens.ndim != 4:
            raise ValueError(
                f"Expected tokens shape (B,N,K,C), got {tuple(tokens.shape)}.",
            )
        if valid.shape != tokens.shape[:3]:
            raise ValueError(
                f"Expected valid shape {tuple(tokens.shape[:3])}, got {tuple(valid.shape)}.",
            )

        if unknown_token is not None:
            unk = unknown_token.to(device=tokens.device, dtype=tokens.dtype)
            tokens_filled = torch.where(valid.unsqueeze(-1), tokens, unk)
            return tokens_filled.mean(dim=-2)

        mask = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        return (tokens * mask).sum(dim=-2) / denom

    @staticmethod
    def _ensure_candidate_batch(candidate_poses_world_cam: PoseTW) -> PoseTW:
        """Ensure candidate poses are batched as ``(B,N,12)``.

        VIN accepts candidates in shape ``(N,12)`` (single batch) or ``(B,N,12)``.
        This helper promotes the unbatched form to ``(1,N,12)`` to simplify
        downstream broadcasting of reference poses and voxel poses.
        """
        if candidate_poses_world_cam.ndim == 2:  # N x 12
            return PoseTW(candidate_poses_world_cam._data.unsqueeze(0))
        return candidate_poses_world_cam

    def _forward_impl(
        self,
        efm: dict[str, Any],
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        return_debug: bool,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinForwardDiagnostics | None]:
        """Run the full VIN forward pass (optionally returning diagnostics).

        Conceptual steps:

        1) **EVL backbone**
           If ``backbone_out`` is not provided, we run EVL once to obtain the
           voxel grid pose ``T_world_voxel`` and evidence volumes (occupancy,
           counts, etc.).

        2) **Candidate pose in reference frame**
           Convert each candidate pose into the reference rig frame:

               T_rig_ref_cam = T_world_rig_ref^{-1} * T_world_cam.

           The candidate center and directions are then:

               t = translation(T_rig_ref_cam)
               r = ||t||
               u = t / (r + eps)
               f = R_rig_ref_cam * z_cam
               s = <f, -u>.

        3) **Pose encoding**
           The tuple (u, f, r, s) is concatenated into ``x = [u, f, r, s]`` and
           encoded with ``LearnableFourierFeatures`` into a fixed-size embedding
           ``pose_enc``.

        4) **Voxel pose encoding (optional)**
           The EVL voxel grid pose is also expressed in the reference rig frame:

               T_rig_ref_voxel = T_world_rig_ref^{-1} * T_world_voxel,

           and encoded with the same LFF pose encoder. This provides a global
           token indicating how the voxel grid is positioned/oriented relative
           to the reference rig.

        5) **Scene field + global token**
           Build the compact voxel field ``F`` from EVL head outputs and project
           it to ``field_dim``. Optionally compute the global mean token:

               global_feat = mean_{x,y,z} F(x,y,z).

        6) **Frustum query (local token)**
           For each candidate camera, build ``K`` frustum points (grid_size^2 *
           len(depths_m)) in world coordinates, map to voxel coordinates, and
           sample ``F`` to obtain ``tokens``. Pool them with a validity mask:

               local_feat = mean_k token_k,

           where invalid samples can be replaced by a learned ``unknown_token``
           (otherwise a masked mean is used).

        7) **Candidate validity + coverage features**
           A candidate is kept if a sufficient fraction of its frustum samples
           lie inside the voxel grid:

               valid_frac = mean_k 1[valid_k],  keep if valid_frac >= min_valid_frac.

           We also optionally concatenate ``voxel_valid_frac`` and ``1 - voxel_valid_frac``
           to expose coverage to the head (low coverage can still imply high RRI).

        8) **Scoring with CORAL**
           Concatenate all tokens and score with the CORAL head. The expected
           ordinal score is:

               E[y] = sum_{k=0}^{K-2} sigmoid(logit_k),
               E_norm = E[y] / (K-1).

        Args:
            efm: Raw EFM snippet dict (unbatched or batched).
            candidate_poses_world_cam: Candidate camera poses as world<-camera.
            reference_pose_world_rig: Reference rig pose (world<-rig) for the snippet.
            p3d_cameras: PyTorch3D cameras aligned with candidates.
            return_debug: Whether to return intermediate tensors for diagnostics.
            backbone_out: Optional precomputed EVL backbone output.

        Returns:
            Tuple of ``(VinPrediction, VinForwardDiagnostics | None)``.
        """
        if backbone_out is None:
            if self.backbone is None:
                raise RuntimeError(
                    "backbone_out is required when the VIN backbone is disabled.",
                )
            backbone_out = self.backbone.forward(efm)
        device = backbone_out.voxel_extent.device
        if next(self.parameters()).device != device:
            self.to(device)
        p3d_cameras = p3d_cameras.to(device)

        pose_world_cam = self._ensure_candidate_batch(candidate_poses_world_cam).to(
            device=device,
        )  # type: ignore[arg-type]
        batch_size, num_candidates = (
            int(pose_world_cam.shape[0]),
            int(pose_world_cam.shape[1]),
        )

        pose_world_rig_ref = reference_pose_world_rig.to(device=device)  # type: ignore[arg-type]
        if pose_world_rig_ref.ndim == 1:
            pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.unsqueeze(0))
        elif pose_world_rig_ref.ndim != 2:
            raise ValueError(
                f"reference_pose_world_rig must have shape (12,) or (B,12), got {pose_world_rig_ref.ndim}",
            )

        if pose_world_rig_ref.shape[0] == 1 and batch_size > 1:
            pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.expand(batch_size, 12))
        elif pose_world_rig_ref.shape[0] != batch_size:
            raise ValueError(
                "reference_pose_world_rig must have batch size 1 or match candidate batch size.",
            )

        # ------------------------------------------------------------------ relative pose (candidate in reference rig frame)
        pose_rig_cam = pose_world_rig_ref.inverse()[:, None] @ pose_world_cam  # rig_ref <- cam

        # ------------------------------------------------------------------ pose encoding (shell descriptor + optional R6D)
        candidate_center_rig_m = pose_rig_cam.t.to(dtype=torch.float32)  # B N 3
        candidate_radius_m = torch.linalg.vector_norm(
            candidate_center_rig_m,
            dim=-1,
            keepdim=True,
        )  # B N 1
        candidate_center_dir_rig = candidate_center_rig_m / (candidate_radius_m + 1e-8)

        cam_forward_axis_cam = torch.tensor(
            [0.0, 0.0, 1.0],
            device=device,
            dtype=torch.float32,
        )
        candidate_forward_dir_rig = torch.einsum(
            "...ij,j->...i",
            pose_rig_cam.R.to(dtype=torch.float32),
            cam_forward_axis_cam,
        )
        candidate_forward_dir_rig = candidate_forward_dir_rig / (
            torch.linalg.vector_norm(candidate_forward_dir_rig, dim=-1, keepdim=True) + 1e-8
        )

        view_alignment = (candidate_forward_dir_rig * (-candidate_center_dir_rig)).sum(
            dim=-1,
            keepdim=True,
        )
        pose_vec: Tensor
        if self.pose_encoding_mode == "t_r6d_lff":
            candidate_rot_r6d = matrix_to_rotation_6d(
                pose_rig_cam.R.to(dtype=torch.float32),
            )
            scales = self._pose_scales()
            pose_vec = torch.cat(
                [
                    candidate_center_rig_m * scales[0],
                    candidate_rot_r6d * scales[1],
                ],
                dim=-1,
            )
        else:
            pose_vec = torch.cat(
                [
                    candidate_center_dir_rig,
                    candidate_forward_dir_rig,
                    candidate_radius_m,
                    view_alignment,
                ],
                dim=-1,
            )
        pose_enc = self.pose_encoder_lff(pose_vec)

        # ------------------------------------------------------------------ voxel pose encoding (reference rig frame)
        voxel_pose_enc: Tensor | None = None
        voxel_center_rig_m: Tensor | None = None
        voxel_radius_m: Tensor | None = None
        voxel_center_dir_rig: Tensor | None = None
        voxel_forward_dir_rig: Tensor | None = None
        voxel_view_alignment: Tensor | None = None

        t_world_voxel = backbone_out.t_world_voxel
        if t_world_voxel.ndim == 1:
            t_world_voxel = PoseTW(t_world_voxel._data.unsqueeze(0))
        if t_world_voxel.shape[0] == 1 and batch_size > 1:
            t_world_voxel = PoseTW(t_world_voxel._data.expand(batch_size, 12))
        elif t_world_voxel.shape[0] != batch_size:
            raise ValueError(
                "voxel/T_world_voxel must have batch size 1 or match candidate batch size.",
            )

        pose_rig_voxel = pose_world_rig_ref.inverse() @ t_world_voxel  # rig_ref <- voxel
        voxel_center_rig_m = pose_rig_voxel.t.to(dtype=torch.float32)  # B 3
        voxel_radius_m = torch.linalg.vector_norm(
            voxel_center_rig_m,
            dim=-1,
            keepdim=True,
        )  # B 1
        voxel_center_dir_rig = voxel_center_rig_m / (voxel_radius_m + 1e-8)
        voxel_forward_dir_rig = torch.einsum(
            "bij,j->bi",
            pose_rig_voxel.R.to(dtype=torch.float32),
            cam_forward_axis_cam,
        )
        voxel_forward_dir_rig = voxel_forward_dir_rig / (
            torch.linalg.vector_norm(voxel_forward_dir_rig, dim=-1, keepdim=True) + 1e-8
        )
        voxel_view_alignment = (voxel_forward_dir_rig * (-voxel_center_dir_rig)).sum(
            dim=-1,
            keepdim=True,
        )
        voxel_pose_vec: Tensor | None = None
        voxel_rot_r6d: Tensor | None = None
        if self.pose_encoding_mode == "t_r6d_lff":
            voxel_rot_r6d = matrix_to_rotation_6d(
                pose_rig_voxel.R.to(dtype=torch.float32),
            )
            scales = self._pose_scales()
            voxel_pose_vec = torch.cat(
                [
                    voxel_center_rig_m * scales[0],
                    voxel_rot_r6d * scales[1],
                ],
                dim=-1,
            )
        else:
            voxel_pose_vec = torch.cat(
                [
                    voxel_center_dir_rig,
                    voxel_forward_dir_rig,
                    voxel_radius_m,
                    voxel_view_alignment,
                ],
                dim=-1,
            )
        voxel_pose_enc = self.pose_encoder_lff.forward(voxel_pose_vec)

        # ------------------------------------------------------------------ build voxel-aligned scene field
        field_in = _build_scene_field(
            backbone_out,
            use_channels=self.config.scene_field_channels,
            occ_input_threshold=self.config.occ_input_threshold,
            counts_norm_mode=self.config.counts_norm_mode,
            occ_pr_is_logits=self.config.occ_pr_is_logits,
        ).to(device=device)
        field = self.field_proj(field_in)

        # ------------------------------------------------------------------ global pooling (coarse tokens)
        parts: list[Tensor] = [pose_enc.to(device=device, dtype=field.dtype)]
        global_feat: Tensor | None = None
        if self.use_global_pool:
            global_feat = self._pool_global(field, pose_enc).to(dtype=field.dtype)
            parts.append(global_feat)
        if self.use_voxel_pose_encoding and voxel_pose_enc is not None:
            voxel_feat = voxel_pose_enc.to(device=device, dtype=field.dtype).unsqueeze(
                1,
            )
            parts.append(voxel_feat.expand(batch_size, num_candidates, -1))

        # ------------------------------------------------------------------ candidate-conditioned frustum query
        points_world = self._frustum_points_world(
            pose_world_cam,
            p3d_cameras=p3d_cameras,
        )
        tokens, token_valid = _sample_voxel_field(
            field,
            points_world=points_world,
            t_world_voxel=backbone_out.t_world_voxel,
            voxel_extent=backbone_out.voxel_extent,
        )
        voxel_valid_frac = token_valid.float().mean(dim=-1, keepdim=True)
        local_feat = self._pool_candidates(
            tokens=tokens,
            valid=token_valid,
            unknown_token=self.unknown_token if self.use_unknown_token else None,
        )
        parts.append(local_feat.to(dtype=field.dtype))
        if self.use_valid_frac_feature:
            parts.append(voxel_valid_frac.to(dtype=field.dtype))
            parts.append((1.0 - voxel_valid_frac).to(dtype=field.dtype))

        # NOTE: Candidate validity is based on the fraction of frustum samples that fall inside the EVL voxel grid
        # (after mapping WORLD→VOXEL using `voxel/T_world_voxel`). We keep this mask for diagnostics and downstream
        # filtering, but do not hard-mask features so low-coverage candidates can still receive high scores.

        candidate_valid = _candidate_valid_from_token(
            token_valid,
            min_valid_frac=self.config.candidate_min_valid_frac,
        )

        feats = torch.cat(parts, dim=-1)
        # feats = feats * candidate_valid.to(dtype=feats.dtype).unsqueeze(-1)
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
            voxel_valid_frac=voxel_valid_frac.squeeze(-1),
            semidense_valid_frac=None,
        )

        if not return_debug:
            return pred, None

        debug = VinForwardDiagnostics(
            backbone_out=backbone_out,
            candidate_center_rig_m=candidate_center_rig_m,
            candidate_radius_m=candidate_radius_m,
            candidate_center_dir_rig=candidate_center_dir_rig,
            candidate_forward_dir_rig=candidate_forward_dir_rig,
            view_alignment=view_alignment,
            pose_enc=pose_enc,
            pose_vec=pose_vec,
            voxel_center_rig_m=voxel_center_rig_m,
            voxel_radius_m=voxel_radius_m,
            voxel_center_dir_rig=voxel_center_dir_rig,
            voxel_forward_dir_rig=voxel_forward_dir_rig,
            voxel_view_alignment=voxel_view_alignment,
            voxel_pose_enc=voxel_pose_enc,
            voxel_pose_vec=voxel_pose_vec,
            field_in=field_in,
            field=field,
            global_feat=global_feat,
            local_feat=local_feat,
            tokens=tokens,
            token_valid=token_valid,
            candidate_valid=candidate_valid,
            voxel_valid_frac=voxel_valid_frac,
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
        """Score candidate poses for one snippet (no diagnostics).

        This is the inference-friendly wrapper around ``_forward_impl``. It
        computes CORAL logits and converts them to:

            prob: class probabilities
            expected: sum_k sigmoid(logit_k)
            expected_normalized: expected / (K-1)

        The output score is *ordinal* rather than metric RRI, but it preserves
        the ordering implied by the learned thresholds.

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: Candidate camera poses as world<-camera.
                Shape can be ``(N,12)`` or ``(B,N,12)``.
            reference_pose_world_rig: Reference rig pose (world<-rig) for the snippet.
            p3d_cameras: PyTorch3D cameras for each candidate (same ordering as candidates).

        Returns:
            :class:`VinPrediction` with CORAL logits, probabilities, and expected scores.
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
        candidate_poses_world_cam: PoseTW,
        reference_pose_world_rig: PoseTW,
        p3d_cameras: PerspectiveCameras,
        backbone_out: EvlBackboneOutput | None = None,
    ) -> tuple[VinPrediction, VinForwardDiagnostics]:
        """Run VIN forward pass and return intermediate tensors.

        This method is intended for debugging and visualization. It exposes the
        *entire* computation graph at key points (pose encoding, scene field
        construction, frustum sampling, masking), enabling detailed sanity
        checks of coordinate transforms and feature magnitudes.

        Args:
            efm: Raw EFM snippet dict.
            candidate_poses_world_cam: Candidate camera poses as world<-camera.
                Shape can be ``(N,12)`` or ``(B,N,12)``.
            reference_pose_world_rig: Reference rig pose (world<-rig) for the snippet.
            p3d_cameras: PyTorch3D cameras for each candidate (same ordering as candidates).

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

    def summarize_vin(
        self,
        batch: VinOracleBatch,
        *,
        include_torchsummary: bool = True,
        torchsummary_depth: int = 3,
    ) -> str:
        """Summarize VIN v1 inputs/outputs for a single oracle-labeled batch."""
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
                "VIN summary requires efm inputs or cached backbone outputs.",
            )

        snippet_view = batch.efm_snippet_view
        efm_dict: dict[str, Any] = {}
        efm_forward: EfmSnippetView | VinSnippetView | dict[str, Any]
        if isinstance(snippet_view, EfmSnippetView):
            efm_dict = snippet_view.efm
            efm_forward = snippet_view
        elif isinstance(snippet_view, VinSnippetView):
            if batch.backbone_out is None:
                raise RuntimeError(
                    "VIN summary requires cached backbone outputs when using VinSnippetView.",
                )
            efm_forward = {}
        else:
            efm_forward = {}

        was_training = self.training
        self.eval()
        with torch.no_grad():
            pred, debug = self.forward_with_debug(
                efm_forward,
                candidate_poses_world_cam=batch.candidate_poses_world_cam,
                reference_pose_world_rig=batch.reference_pose_world_rig,
                p3d_cameras=batch.p3d_cameras,
                backbone_out=batch.backbone_out,
            )
        if was_training:
            self.train()
        backbone_out = debug.backbone_out
        if snippet_view is None:
            efm_summary = {"note": "cached batch (raw EFM inputs unavailable)"}
        elif isinstance(snippet_view, VinSnippetView):
            efm_summary = {
                "note": "VIN snippet cache (no raw EFM inputs)",
                "vin_snippet.points_world": summarize(snippet_view.points_world),
                "vin_snippet.lengths": summarize(snippet_view.lengths),
                "vin_snippet.t_world_rig": summarize(snippet_view.t_world_rig.tensor()),
            }
        else:
            efm_summary = {
                **{key: summarize(efm_dict.get(key)) for key in ARIA_IMG},
                **{key: summarize(efm_dict.get(key)) for key in ARIA_CALIB},
                ARIA_POSE_T_WORLD_RIG: summarize(efm_dict.get(ARIA_POSE_T_WORLD_RIG)),
            }

        summary_dict: dict[str, Any] = {
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
                "candidate_radius_m": summarize(
                    debug.candidate_radius_m,
                    include_stats=True,
                ),
                "candidate_center_dir_rig": summarize(
                    debug.candidate_center_dir_rig,
                    include_stats=True,
                ),
                "candidate_forward_dir_rig": summarize(
                    debug.candidate_forward_dir_rig,
                    include_stats=True,
                ),
                "view_alignment": summarize(debug.view_alignment, include_stats=True),
                "pose_vec": summarize(debug.pose_vec, include_stats=True),
                "pose_enc": summarize(debug.pose_enc),
            },
            "features": {
                "field_in": summarize(debug.field_in),
                "field": summarize(debug.field),
                "global_feat": summarize(debug.global_feat),
                "local_feat": summarize(debug.local_feat),
                "tokens": summarize(debug.tokens),
                "token_valid": summarize(debug.token_valid),
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
            if efm_dict and key in efm_dict:
                summary_dict.setdefault("efm", {})[key] = summarize(efm_dict.get(key))

        voxel_pose = {
            "voxel_center_rig_m": summarize(
                debug.voxel_center_rig_m,
                include_stats=True,
            ),
            "voxel_radius_m": summarize(debug.voxel_radius_m, include_stats=True),
            "voxel_center_dir_rig": summarize(
                debug.voxel_center_dir_rig,
                include_stats=True,
            ),
            "voxel_forward_dir_rig": summarize(
                debug.voxel_forward_dir_rig,
                include_stats=True,
            ),
            "voxel_view_alignment": summarize(
                debug.voxel_view_alignment,
                include_stats=True,
            ),
            "voxel_pose_vec": summarize(debug.voxel_pose_vec, include_stats=True),
            "voxel_pose_enc": summarize(debug.voxel_pose_enc),
        }
        if any(value is not None for value in voxel_pose.values()):
            summary_dict["voxel_pose"] = voxel_pose

        tree = rich_summary(
            tree_dict=summary_dict,
            root_label="VIN v1 summary (oracle batch)",
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

            pose_vec = debug.pose_vec
            feats_2d = debug.feats.reshape(
                debug.feats.shape[0] * debug.feats.shape[1],
                -1,
            )

            if pose_vec is not None:
                pose_vec_2d = pose_vec.reshape(
                    pose_vec.shape[0] * pose_vec.shape[1],
                    -1,
                )
                lines.append("torchsummary: pose_encoder_lff (trainable)")
                lines.append(
                    str(
                        torch_summary(
                            self.pose_encoder_lff,
                            input_data=pose_vec_2d,
                            verbose=0,
                            depth=torchsummary_depth,
                            device=debug.candidate_center_rig_m.device,
                        ),
                    ),
                )
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

            lines.append("torchsummary: VinScorerHead (trainable)")
            lines.append(
                str(
                    torch_summary(
                        self.head,
                        input_data=feats_2d,
                        verbose=0,
                        depth=torchsummary_depth,
                        device=debug.candidate_center_rig_m.device,
                    ),
                ),
            )

        return "\n".join(lines)

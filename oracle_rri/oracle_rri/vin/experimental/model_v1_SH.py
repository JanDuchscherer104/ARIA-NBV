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

1. **Shell pose encoding** (candidate pose in reference frame)

   Given the relative pose:

       T_rig_ref_cam = T_world_rig_ref^{-1} * T_world_cam,

   we define:

       t = translation(T_rig_ref_cam)        (candidate center in rig-ref),
       r = ||t||                               (radius),
       u = t / (r + eps)                       (center direction),
       f = R_rig_ref_cam * z_cam               (camera forward direction),
       s = <f, -u>                             (view alignment scalar).

   These are encoded by ``ShellShPoseEncoder`` using real spherical harmonics for
   ``u`` and ``f`` plus Fourier features for ``r`` and an MLP for ``s``.

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

       [pose_enc, global_field_mean?, voxel_pose_enc?, local_frustum_feat].

   The head outputs CORAL logits ``l_k`` for thresholds ``k=0..K-2``:

       P(y > k) = sigmoid(l_k),
       E[y] = sum_k P(y > k),
       E_norm = E[y] / (K-1).

These steps provide an interpretable mapping from candidate pose + EVL scene
context to an ordinal NBV score that correlates with oracle RRI.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from pydantic import Field, field_validator
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
from torch import Tensor, nn

from ...rri_metrics.coral import CoralLayer, coral_expected_from_logits, coral_logits_to_prob
from ...utils import BaseConfig
from ..backbone_evl import EvlBackboneConfig
from .spherical_encoding import ShellShPoseEncoderConfig
from .types import EvlBackboneOutput, VinForwardDiagnostics, VinPrediction


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

    samp, valid_grid = sample_voxels(field, pts_vox_id, differentiable=False)  # B C (N*K), B (N*K)
    valid = (valid_extent & valid_grid).reshape(batch_size, num_candidates, num_points)
    tokens = samp.transpose(1, 2).reshape(batch_size, num_candidates, num_points, field_channels)
    return tokens, valid


def _candidate_valid_from_token(token_valid: Tensor, *, min_valid_frac: float) -> Tensor:
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
        raise ValueError(f"Expected token_valid with ndim>=1, got {tuple(token_valid.shape)}.")
    valid_frac = token_valid.float().mean(dim=-1)
    return valid_frac >= min_valid_frac


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

    def __init__(self, config: "VinScorerHeadConfig", *, in_dim: int | None = None) -> None:
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
        return self.target(self, in_dim=in_dim)


class VinModelConfig(BaseConfig["VinModel"]):
    """Configuration for :class:`VinModel`.

    This config collects all architectural choices that determine how VIN
    represents scene context and candidate poses. Conceptually, the VIN score is
    a function:

        score = f( pose_enc(u,f,r,s),
                   voxel_pose_enc?,
                   global_field_mean?,
                   local_frustum_feat ),

    where ``pose_enc`` and ``voxel_pose_enc`` are SH-based shell encodings,
    ``global_field_mean`` summarizes the voxel field, and ``local_frustum_feat``
    samples the voxel field along a candidate frustum.
    """

    target: type["VinModel"] = Field(default_factory=lambda: VinModel, exclude=True)
    """Factory target for :meth:`BaseConfig.setup_target`."""

    backbone: EvlBackboneConfig = Field(default_factory=EvlBackboneConfig)
    """Frozen EVL backbone configuration."""

    pose_encoder_sh: ShellShPoseEncoderConfig = Field(default_factory=ShellShPoseEncoderConfig)
    """Spherical harmonics pose encoding configuration (shell descriptor)."""

    head: VinScorerHeadConfig = Field(default_factory=VinScorerHeadConfig)
    """Scoring head configuration."""

    scene_field_channels: Literal[
        "occ_pr", "occ_input", "counts_norm", "observed", "unknown", "new_surface_prior", "free_input"
    ] = Field(
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

    use_global_pool: bool = True
    """Whether to concatenate the global mean-pooled embedding to per-candidate features."""

    use_voxel_pose_encoding: bool = True
    """Whether to append a SH-encoded voxel-grid pose (voxel/T_world_voxel) in the reference frame."""

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
        """Validate frustum depths.

        Depths are used as *metric z* in camera space before unprojection, so
        they must be finite and strictly positive.
        """
        bad = [d for d in value if (not math.isfinite(d)) or d <= 0.0]
        if bad:
            raise ValueError(f"frustum_depths_m must contain finite values > 0, got {bad}")
        return value


class VinModel(nn.Module):
    """View Introspection Network (VIN) predicting RRI from EVL voxel features + pose.

    VIN is a light-weight head that queries frozen EVL voxel features to score
    candidate camera poses. The architecture is deliberately simple:

    - **Pose encoding** via real spherical harmonics (direction) and Fourier
      features (radius) to represent candidate shells.
    - **Scene field** built from EVL evidence volumes and projected with a
      1x1x1 Conv3d to a small feature dimension.
    - **Local query**: sample the scene field at frustum points and pool.
    - **Global tokens**: optional mean-pooled field + optional voxel-pose token.
    - **CORAL head** to produce ordinal scores.

    The overall score is computed as:

        z = concat(pose_enc, global_field_mean?, voxel_pose_enc?, local_feat)
        logits = CORAL(MLP(z))
        score = E[y]/(K-1) = (1/(K-1)) * sum_k sigmoid(logit_k)
    """

    def __init__(self, config: VinModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = self.config.backbone.setup_target()
        self.pose_encoder_sh = self.config.pose_encoder_sh.setup_target()

        field_dim = self.config.field_dim
        gn_groups = _largest_divisor_leq(field_dim, self.config.field_gn_groups)

        field_in_dim = len(self.config.scene_field_channels)
        self.field_proj = nn.Sequential(
            nn.Conv3d(field_in_dim, field_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=field_dim),
            nn.GELU(),
        )

        self.use_global_pool = self.config.use_global_pool
        self.use_voxel_pose_encoding = self.config.use_voxel_pose_encoding

        # Head input dim is data-dependent (feature channel count depends on EVL cfg).
        pose_dim = int(self.pose_encoder_sh.out_dim)
        head_in_dim = pose_dim + field_dim
        if self.use_global_pool:
            head_in_dim += field_dim
        if self.use_voxel_pose_encoding:
            head_in_dim += pose_dim
        self.head = self.config.head.setup_target(in_dim=head_in_dim)
        self.to(self.backbone.device)

    def _pool_global(self, field: Tensor) -> Tensor:
        """Mean-pool global context from a voxel field.

        This produces a scene-level token:

            global_feat = mean_{x,y,z} F(x,y,z).

        The global token provides coarse context such as overall occupancy
        density and semantic bias across the snippet.
        """

        return field.mean(dim=(-3, -2, -1))

    def _frustum_points_world(self, poses_world_cam: PoseTW, *, p3d_cameras: PerspectiveCameras) -> Tensor:
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
                "poses_world_cam must have shape (B,N,12). Use `_ensure_candidate_batch` before calling this helper."
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
            f"got {num_cams} for B={batch_size}, N={num_candidates}."
        )

    def _pool_candidates(self, *, tokens: Tensor, valid: Tensor) -> Tensor:
        """Mean-pool candidate-local frustum samples.

        For each candidate, we compute a masked mean over K frustum samples:

            local_feat = sum_k (valid_k * token_k) / (sum_k valid_k + eps).

        This aggregates the local voxel evidence along the candidate frustum
        while ignoring samples that fall outside the voxel grid.
        """

        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens shape (B,N,K,C), got {tuple(tokens.shape)}.")
        if valid.shape != tokens.shape[:3]:
            raise ValueError(f"Expected valid shape {tuple(tokens.shape[:3])}, got {tuple(valid.shape)}.")

        mask = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        pooled = (tokens * mask).sum(dim=-2) / denom
        return pooled

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
           The tuple (u, f, r, s) is encoded with ``ShellShPoseEncoder`` into a
           fixed-size embedding ``pose_enc``.

        4) **Voxel pose encoding (optional)**
           The EVL voxel grid pose is also expressed in the reference rig frame:

               T_rig_ref_voxel = T_world_rig_ref^{-1} * T_world_voxel,

           and encoded with the same shell encoder. This provides a global token
           indicating how the voxel grid is positioned/oriented relative to the
           reference rig.

        5) **Scene field + global token**
           Build the compact voxel field ``F`` from EVL head outputs and project
           it to ``field_dim``. Optionally compute the global mean token:

               global_feat = mean_{x,y,z} F(x,y,z).

        6) **Frustum query (local token)**
           For each candidate camera, build ``K`` frustum points (grid_size^2 *
           len(depths_m)) in world coordinates, map to voxel coordinates, and
           sample ``F`` to obtain ``tokens``. Pool them with a validity mask:

               local_feat = sum_k (valid_k * token_k) / (sum_k valid_k + eps).

        7) **Candidate validity**
           A candidate is kept if a sufficient fraction of its frustum samples
           lie inside the voxel grid:

               valid_frac = mean_k 1[valid_k],  keep if valid_frac >= min_valid_frac.

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
            backbone_out = self.backbone.forward(efm)
        device = backbone_out.voxel_extent.device

        pose_world_cam = self._ensure_candidate_batch(candidate_poses_world_cam).to(device=device)  # type: ignore[arg-type]
        batch_size, num_candidates = int(pose_world_cam.shape[0]), int(pose_world_cam.shape[1])

        pose_world_rig_ref = reference_pose_world_rig.to(device=device)  # type: ignore[arg-type]
        if pose_world_rig_ref.ndim == 1:
            pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.unsqueeze(0))
        elif pose_world_rig_ref.ndim != 2:
            raise ValueError(f"reference_pose_world_rig must have shape (12,) or (B,12), got {pose_world_rig_ref.ndim}")

        if pose_world_rig_ref.shape[0] == 1 and batch_size > 1:
            pose_world_rig_ref = PoseTW(pose_world_rig_ref._data.expand(batch_size, 12))
        elif pose_world_rig_ref.shape[0] != batch_size:
            raise ValueError("reference_pose_world_rig must have batch size 1 or match candidate batch size.")

        # ------------------------------------------------------------------ relative pose (candidate in reference rig frame)
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
            raise ValueError("voxel/T_world_voxel must have batch size 1 or match candidate batch size.")

        pose_rig_voxel = pose_world_rig_ref.inverse() @ t_world_voxel  # rig_ref <- voxel
        voxel_center_rig_m = pose_rig_voxel.t.to(dtype=torch.float32)  # B 3
        voxel_radius_m = torch.linalg.vector_norm(voxel_center_rig_m, dim=-1, keepdim=True)  # B 1
        voxel_center_dir_rig = voxel_center_rig_m / (voxel_radius_m + 1e-8)
        voxel_forward_dir_rig = torch.einsum(
            "bij,j->bi", pose_rig_voxel.R.to(dtype=torch.float32), cam_forward_axis_cam
        )
        voxel_forward_dir_rig = voxel_forward_dir_rig / (
            torch.linalg.vector_norm(voxel_forward_dir_rig, dim=-1, keepdim=True) + 1e-8
        )
        voxel_view_alignment = (voxel_forward_dir_rig * (-voxel_center_dir_rig)).sum(dim=-1, keepdim=True)
        voxel_pose_enc = self.pose_encoder_sh.forward(
            voxel_center_dir_rig,
            voxel_forward_dir_rig,
            r=voxel_radius_m,
            scalars=voxel_view_alignment,
        )

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
            global_feat = self._pool_global(field).unsqueeze(1).expand(batch_size, num_candidates, -1)
            parts.append(global_feat)
        if self.use_voxel_pose_encoding and voxel_pose_enc is not None:
            voxel_feat = voxel_pose_enc.to(device=device, dtype=field.dtype).unsqueeze(1)
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
        local_feat = self._pool_candidates(tokens=tokens, valid=token_valid)
        parts.append(local_feat.to(dtype=field.dtype))

        # NOTE: Candidate validity is based on the fraction of frustum samples that fall inside the EVL voxel grid
        # (after mapping WORLD→VOXEL using `voxel/T_world_voxel`). This avoids admitting candidates with only a
        # handful of in-bounds samples.
        candidate_valid = _candidate_valid_from_token(
            token_valid,
            min_valid_frac=self.config.candidate_min_valid_frac,
        )
        voxel_valid_frac = token_valid.float().mean(dim=-1, keepdim=True)

        feats = torch.cat(parts, dim=-1)
        feats = feats * candidate_valid.to(dtype=feats.dtype).unsqueeze(-1)
        logits = self.head(feats.reshape(batch_size * num_candidates, -1)).reshape(batch_size, num_candidates, -1)

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
            voxel_center_rig_m=voxel_center_rig_m,
            voxel_radius_m=voxel_radius_m,
            voxel_center_dir_rig=voxel_center_dir_rig,
            voxel_forward_dir_rig=voxel_forward_dir_rig,
            voxel_view_alignment=voxel_view_alignment,
            voxel_pose_enc=voxel_pose_enc,
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

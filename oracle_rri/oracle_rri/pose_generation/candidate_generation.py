"""Candidate pose generation with modular sampling and pruning rules.

This module implements a clear three-stage pipeline:

1. **Sample** directions + radii around the latest pose using either
   area-uniform or forward-biased spherical distributions in the rig frame.
2. **Construct** roll-free camera poses that look away from the last pose
   while keeping the camera x-axis horizontal in the world frame.
3. **Prune** candidates via rule objects (mesh clearance, collision, free
   space). Rules may optionally emit diagnostics such as per-candidate mesh
   distances.

All poses are expressed as :class:`efm3d.aria.pose.PoseTW` in the VIO world
frame (LUF camera convention: x=left, y=up, z=forward).
"""

from __future__ import annotations

from math import ceil
from typing import Literal

import torch
import trimesh
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from power_spherical import HypersphericalUniform, PowerSpherical
from pydantic import AliasChoices, Field, field_validator, model_validator

from ..data.efm_views import EfmSnippetView
from ..data.mesh_cache import mesh_from_snippet
from ..utils import BaseConfig, Console, Verbosity, select_device
from ..utils.frames import view_axes_from_poses, world_up_tensor
from .candidate_generation_rules import FreeSpaceRule, MinDistanceToMeshRule, PathCollisionRule, Rule
from .types import (
    CandidateContext,
    CandidateSamplingResult,
    CollisionBackend,
    SamplingStrategy,
    ViewDirectionMode,
)

DEVICE_FWD = [0.0, 0.0, 1.0]


class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    """Config for candidate generation around the latest pose.

    Sampling
    --------
    num_samples:
        Number of candidate poses requested *after* pruning.
    oversample_factor:
        Factor for initial oversampling before pruning to offset rule
        rejections.
    max_resamples:
        Maximum number of oversampling rounds if pruning removes too many
        candidates.
    min_radius / max_radius:
        Inner and outer radii (metres) of the sampling shell around the last
        pose.
    min_elev_deg / max_elev_deg:
        Elevation band in degrees relative to the world horizontal plane.
    delta_azimuth_deg:
        Horizontal yaw band in degrees around the last forward direction.
        360 → unrestricted, 90 → ±45° about forward.
    sampling_strategy:
        Distribution for direction sampling in rig frame
        (:class:`SamplingStrategy`).
    kappa:
        Concentration for the forward-biased PowerSpherical sampler.

    Rules
    -----
    min_distance_to_mesh:
        Clearance enforced between camera centre and mesh (metres).
    ensure_collision_free:
        Enable straight-line path collision filtering.
    ensure_free_space:
        Enable AABB workspace filtering.
    collision_backend:
        Backend for collision / distance checks (``P3D``, ``PYEMBREE``, or
        ``TRIMESH``).
    ray_subsample:
        Number of samples along a ray when using discretised collision tests.
    step_clearance:
        Distance threshold (metres) for discretised collision rejection.

    Diagnostics
    -----------
    collect_rule_masks:
        When True, store per-rule boolean masks in the result for analysis.
    collect_debug_stats:
        When True, allow rules to emit extra tensors (e.g., distances to
        mesh) in ``CandidateSamplingResult.extras``.
    """

    target: type["CandidateViewGenerator"] = Field(default_factory=lambda: CandidateViewGenerator, exclude=True)

    camera_label: Literal["rgb", "slaml", "slamr"] = "rgb"
    """Camera index to use for candidate generation."""

    num_samples: int = 512
    oversample_factor: float = 2.0
    max_resamples: int = 4

    min_radius: float = 0.4
    max_radius: float = 1.6

    min_elev_deg: float = -15.0
    max_elev_deg: float = 45.0
    delta_azimuth_deg: float = 360.0

    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM_SPHERE
    kappa: float = 4.0

    min_distance_to_mesh: float = 0.0
    ensure_collision_free: bool = True
    ensure_free_space: bool = True
    collision_backend: CollisionBackend = CollisionBackend.P3D
    ray_subsample: int = 128
    step_clearance: float = 0.05

    mesh_samples: int | None = None

    device: torch.device = Field(default_factory=lambda: select_device("auto", component="CandidateViewGenerator"))
    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging.",
    )
    is_debug: bool = False

    collect_rule_masks: bool = False
    collect_debug_stats: bool = False

    reference_frame_index: int | None = None
    """Optional camera frame index to use as reference pose; None defaults to the final pose."""

    # View orientation controls
    view_direction_mode: ViewDirectionMode = ViewDirectionMode.RADIAL_AWAY
    """Base orientation strategy for candidates."""

    view_sampling_strategy: SamplingStrategy | None = None
    """View-direction sampling in base camera frame; None disables view jitter."""

    view_kappa: float | None = None
    """Concentration for PowerSpherical view sampler; defaults to positional ``kappa`` when None."""

    view_max_angle_deg: float = 0.0
    """Optional cap on angular deviation (deg) from forward in the base camera frame."""

    view_roll_jitter_deg: float = 0.0
    """Symmetric roll jitter (deg) around the sampled forward axis in camera frame."""

    view_target_point_world: torch.Tensor | None = None
    """Optional world-space target for TARGET_POINT mode (shape (3,))."""

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, value: str | torch.device) -> torch.device:
        return select_device(value, component="CandidateViewGenerator")

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Verbosity | int | str) -> Verbosity:
        return Verbosity.from_any(value)

    @model_validator(mode="after")
    def set_debug(self) -> "CandidateViewGeneratorConfig":
        if self.is_debug:
            object.__setattr__(self, "device", torch.device("cpu"))
            object.__setattr__(self, "verbosity", Verbosity.VERBOSE)
        if self.view_kappa is None:
            object.__setattr__(self, "view_kappa", self.kappa)
        return self


# ---------------------------------------------------------------------------
# Direction samplers
# ---------------------------------------------------------------------------


class DirectionSampler:
    """Abstract base class for unit direction sampling in rig (LUF) frame."""

    name: str = "base"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class UniformDirectionSampler(DirectionSampler):
    name = "uniform"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        dist = HypersphericalUniform(dim=3, device=device)
        dirs = dist.sample((num,))
        return dirs / dirs.norm(dim=-1, keepdim=True)


class ForwardPowerSphericalSampler(DirectionSampler):
    name = "forward_power_spherical"

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        mu = torch.tensor(DEVICE_FWD, device=device)
        dist = PowerSpherical(mu, torch.tensor(cfg.kappa, device=device))
        dirs = dist.sample((num,))
        return dirs / dirs.norm(dim=-1, keepdim=True)


_DIRECTION_SAMPLERS: dict[SamplingStrategy, DirectionSampler] = {
    SamplingStrategy.UNIFORM_SPHERE: UniformDirectionSampler(),
    SamplingStrategy.FORWARD_POWERSPHERICAL: ForwardPowerSphericalSampler(),
}


# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------


def _forward_world(reference_pose: PoseTW) -> torch.Tensor:
    """World-frame forward direction (+z_cam) of the reference pose."""

    f_cam = torch.tensor(DEVICE_FWD, device=reference_pose.device, dtype=reference_pose.dtype)
    return reference_pose.rotate(f_cam.unsqueeze(0))[0]


def _ensure_unbatched_pose(pose: PoseTW) -> PoseTW:
    """Ensure PoseTW has shape (12,) instead of (1,12) when singleton batch."""

    if pose._data.ndim == 2 and pose._data.shape[0] == 1:
        return PoseTW(pose._data.squeeze(0))
    return pose


def _filter_directions_world(
    dirs_world: torch.Tensor,
    last_forward_world: torch.Tensor,
    cfg: CandidateViewGeneratorConfig,
) -> torch.Tensor:
    """Filter directions by elevation and delta-azimuth in world frame.

    Returns a boolean mask of shape ``(N,)``.
    """

    device = dirs_world.device
    wup = world_up_tensor(device=device, dtype=dirs_world.dtype)

    # Elevation relative to horizontal plane
    dot_up = (dirs_world * wup).sum(dim=-1)
    elev = torch.asin(dot_up.clamp(-1.0, 1.0))

    min_elev = torch.deg2rad(torch.tensor(cfg.min_elev_deg, device=device, dtype=dirs_world.dtype))
    max_elev = torch.deg2rad(torch.tensor(cfg.max_elev_deg, device=device, dtype=dirs_world.dtype))
    mask_elev = (elev >= min_elev) & (elev <= max_elev)

    # Yaw around world-up relative to last forward
    def _project_horizontal(v: torch.Tensor) -> torch.Tensor:
        dot = (v * wup).sum(dim=-1, keepdim=True)
        v_h = v - dot * wup
        return v_h / v_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dirs_h = _project_horizontal(dirs_world)
    fwd_h = _project_horizontal(last_forward_world.view(1, 3)).expand_as(dirs_h)

    cross = torch.cross(fwd_h, dirs_h, dim=-1)
    sin_yaw = (cross * wup).sum(dim=-1)
    cos_yaw = (dirs_h * fwd_h).sum(dim=-1)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if cfg.delta_azimuth_deg >= 360.0 - 1e-1:
        mask_yaw = torch.ones_like(mask_elev)
    else:
        half_delta = 0.5 * torch.deg2rad(torch.tensor(cfg.delta_azimuth_deg, device=device, dtype=dirs_world.dtype))
        mask_yaw = (yaw >= -half_delta) & (yaw <= half_delta)

    return mask_elev & mask_yaw


def _sample_candidate_positions(
    reference_pose: PoseTW,
    cfg: CandidateViewGeneratorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate centres around the last pose.

    Returns:
        centers_world: (N, 3) camera centres in world frame.
        offsets_ref: (N, 3) sampled directions in reference frame (before radius).
    """

    device = cfg.device
    sampler = _DIRECTION_SAMPLERS[cfg.sampling_strategy]

    reference_pose_dev = _ensure_unbatched_pose(reference_pose.to(device))
    fwd_world = _forward_world(reference_pose_dev)

    dirs_world_list: list[torch.Tensor] = []
    offsets_rig_list: list[torch.Tensor] = []

    remaining = cfg.num_samples
    rounds = 0

    while remaining > 0 and rounds < cfg.max_resamples:
        rounds += 1
        n_draw = ceil(cfg.oversample_factor * remaining)

        dirs_rig = sampler.sample(cfg, n_draw, device=device)
        dirs_world = reference_pose_dev.rotate(dirs_rig)
        mask = _filter_directions_world(dirs_world, fwd_world, cfg).view(-1)

        if mask.any():
            dirs_world_list.append(dirs_world[mask])
            offsets_rig_list.append(dirs_rig[mask])
            remaining = cfg.num_samples - sum(d.shape[0] for d in dirs_world_list)

    if not dirs_world_list:
        raise RuntimeError("Directional sampling failed; relax elevation/azimuth constraints.")
    dirs_world_all = torch.cat(dirs_world_list, dim=0).view(-1, 3)
    offsets_rig_all = torch.cat(offsets_rig_list, dim=0).view(-1, 3)

    if dirs_world_all.shape[0] < cfg.num_samples:
        deficit = cfg.num_samples - dirs_world_all.shape[0]
        dirs_world_all = torch.cat([dirs_world_all, dirs_world_all[:deficit]], dim=0)
        offsets_rig_all = torch.cat([offsets_rig_all, offsets_rig_all[:deficit]], dim=0)

    dirs_world_all = dirs_world_all[: cfg.num_samples]
    offsets_rig_all = offsets_rig_all[: cfg.num_samples]

    radii = torch.empty(dirs_world_all.shape[0], device=device, dtype=dirs_world_all.dtype).uniform_(
        cfg.min_radius, cfg.max_radius
    )
    offsets_rig_all = offsets_rig_all * radii[:, None]
    centers_world = reference_pose_dev.transform(offsets_rig_all)
    return centers_world, offsets_rig_all


def _sample_view_dirs_cam(
    cfg: CandidateViewGeneratorConfig,
    num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample view directions on S² in the base camera frame."""

    strat = cfg.view_sampling_strategy
    if strat is None:
        v = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
        return v.view(1, 3).expand(num, 3)

    if strat == SamplingStrategy.UNIFORM_SPHERE:
        dist = HypersphericalUniform(dim=3, device=device, dtype=dtype)
    elif strat == SamplingStrategy.FORWARD_POWERSPHERICAL:
        mu = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
        scale = torch.tensor(cfg.view_kappa, device=device, dtype=dtype)
        dist = PowerSpherical(mu, scale)
    else:
        raise ValueError(f"Unsupported view_sampling_strategy: {strat}")

    dirs = dist.rsample((num,))
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    if cfg.view_max_angle_deg > 0.0:
        mu = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
        cos_max = torch.cos(
            torch.tensor(torch.deg2rad(torch.tensor(cfg.view_max_angle_deg)), device=device, dtype=dtype)
        )
        mask = (dirs * mu).sum(dim=-1) < cos_max
        tries = 0
        while mask.any() and tries < 8:
            tries += 1
            resample_n = int(mask.sum().item())
            new_dirs = dist.rsample((resample_n,))
            new_dirs = new_dirs / new_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dirs[mask] = new_dirs
            mask = (dirs * mu).sum(dim=-1) < cos_max

    return dirs


def _build_candidate_orientations(
    reference_pose: PoseTW,
    centers_world: torch.Tensor,
    cfg: CandidateViewGeneratorConfig,
) -> PoseTW:
    """Construct candidate orientations from centres and view settings."""

    device = centers_world.device
    dtype = centers_world.dtype
    n = centers_world.shape[0]

    reference_pose_dev = _ensure_unbatched_pose(reference_pose.to(device))

    # Base orientations
    if cfg.view_direction_mode is ViewDirectionMode.FORWARD_RIG:
        r_last = reference_pose_dev.R
        if r_last.ndim == 3:
            r_last = r_last[0]
        r_base = r_last.unsqueeze(0).expand(n, 3, 3)
        base_poses = PoseTW.from_Rt(r_base, centers_world)

    elif cfg.view_direction_mode in (ViewDirectionMode.RADIAL_AWAY, ViewDirectionMode.RADIAL_TOWARDS):
        eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
        centers_pose = PoseTW.from_Rt(eye, centers_world)
        base_poses = view_axes_from_poses(
            from_pose=reference_pose_dev,
            to_pose=centers_pose,
            look_away=(cfg.view_direction_mode is ViewDirectionMode.RADIAL_AWAY),
        )

    elif cfg.view_direction_mode is ViewDirectionMode.TARGET_POINT:
        if cfg.view_target_point_world is None:
            raise ValueError("TARGET_POINT mode requires `view_target_point_world` to be set.")
        target = cfg.view_target_point_world.to(device=device, dtype=dtype).view(1, 3)
        wup = world_up_tensor(device=device, dtype=dtype)
        v = target - centers_world
        z_world = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dot_up = (z_world * wup.view(1, 3)).sum(dim=-1, keepdim=True)
        y_world = wup.view(1, 3) - dot_up * z_world
        y_world = y_world / y_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_world = torch.cross(y_world, z_world, dim=-1)
        r_base = torch.stack([x_world, y_world, z_world], dim=-1)
        base_poses = PoseTW.from_Rt(r_base, centers_world)

    else:
        raise ValueError(f"Unsupported view_direction_mode: {cfg.view_direction_mode}")

    if cfg.view_sampling_strategy is None and cfg.view_roll_jitter_deg == 0.0:
        return base_poses

    dirs_cam = _sample_view_dirs_cam(cfg, n, device=device, dtype=dtype)
    z_new = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    up_cam = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand_as(z_new)
    x_new = torch.cross(up_cam, z_new, dim=-1)
    x_new = x_new / x_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y_new = torch.cross(z_new, x_new, dim=-1)
    y_new = y_new / y_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    r_delta = torch.stack([x_new, y_new, z_new], dim=-1)

    if cfg.view_roll_jitter_deg > 0.0:
        roll = (2.0 * torch.rand(n, device=device, dtype=dtype) - 1.0) * torch.deg2rad(
            torch.tensor(cfg.view_roll_jitter_deg, device=device, dtype=dtype)
        )
        cr, sr = torch.cos(roll), torch.sin(roll)
        r_roll = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        r_roll[:, 0, 0] = cr
        r_roll[:, 0, 1] = -sr
        r_roll[:, 1, 0] = sr
        r_roll[:, 1, 1] = cr
        r_roll[:, 2, 2] = 1.0
        r_delta = torch.matmul(r_delta, r_roll)

    delta_poses = PoseTW.from_Rt(r_delta, torch.zeros_like(centers_world))
    return base_poses.compose(delta_poses)


class CandidateViewGenerator:
    """Generate candidate :class:`PoseTW` around the latest pose using rules."""

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self._rules: list[Rule] = self._build_default_rules(config)

    # ------------------------------------------------------------------ public
    def generate_from_typed_sample(
        self, sample: EfmSnippetView, frame_index: int | None = None
    ) -> CandidateSamplingResult:
        """Generate candidates using an :class:`EfmSnippetView` sample.

        Args:
            sample: Snippet view with trajectory and mesh.
            frame_index: Optional frame index to extract the reference pose instead of using the final pose.
                0 <= frame_index < F where is the number of frames in the snippet; F = sample.get_camera(self.config.camera_label).num_frames.
        """

        device = torch.device(self.config.device)
        occ = sample.get_occupancy_extend()
        self.console.dbg(f"Using occupancy extent: [xmin, xmax, ymin, ymax, zmin, zmax] = {occ}")
        occupancy_extent = occ.to(device=device, dtype=torch.float32)
        gt_mesh = sample.mesh
        mesh_verts = sample.mesh_verts
        mesh_faces = sample.mesh_faces
        if gt_mesh is not None and (mesh_verts is None or mesh_faces is None):
            artifact = mesh_from_snippet(sample, device=device, console=self.console)
            gt_mesh = artifact.processed.mesh
            mesh_verts = artifact.processed.verts
            mesh_faces = artifact.processed.faces

        assert mesh_verts is not None and mesh_faces is not None, "Mesh vertices and faces must be provided."

        cam_view = sample.get_camera(self.config.camera_label)

        if frame_index is None:
            frame_index = self.config.reference_frame_index

        if frame_index is None:
            reference_pose = sample.trajectory.final_pose.to(device=device)
        else:
            cam_idx, traj_idx = cam_view.nearest_traj_indices(
                sample.trajectory.time_ns, [frame_index], default_last=True
            )
            if traj_idx.numel() == 0:
                reference_pose = sample.trajectory.final_pose.to(device=device)
            else:
                reference_pose = sample.trajectory.t_world_rig[traj_idx].to(device=device)

        return self.generate(
            reference_pose=reference_pose,
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            camera_calib_template=cam_view.calib,
            occupancy_extent=occupancy_extent,
        )

    def generate(
        self,
        *,
        reference_pose: PoseTW,
        gt_mesh: trimesh.Trimesh,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        camera_calib_template: CameraTW,
        occupancy_extent: torch.Tensor,
    ) -> CandidateSamplingResult:
        """Sample candidate poses around ``reference_pose`` and apply pruning rules."""

        cfg = self.config
        device = torch.device(cfg.device)

        centers_world, offsets_ref = _sample_candidate_positions(reference_pose, cfg)
        shell_poses = _build_candidate_orientations(reference_pose, centers_world, cfg)

        ctx = CandidateContext(
            cfg=cfg,
            reference_pose=reference_pose.to(device),
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts.to(device),
            mesh_faces=mesh_faces.to(device),
            occupancy_extent=occupancy_extent.to(device),
            camera_calib_template=camera_calib_template.to(device),
            shell_poses=shell_poses,
            centers_world=centers_world,
            shell_offsets_ref=offsets_ref,
            mask_valid=torch.ones(centers_world.shape[0], dtype=torch.bool, device=device),
        )

        self._apply_rules(ctx)

        return self._finalise(ctx)

    # ------------------------------------------------------------------ internals
    def _build_default_rules(self, cfg: CandidateViewGeneratorConfig) -> list[Rule]:
        rules: list[Rule] = []
        if cfg.ensure_free_space:
            rules.append(FreeSpaceRule(cfg))
        if cfg.min_distance_to_mesh > 0:
            rules.append(MinDistanceToMeshRule(cfg))
        if cfg.ensure_collision_free:
            rules.append(PathCollisionRule(cfg))
        return rules

    def _apply_rules(self, ctx: CandidateContext) -> None:
        for rule in self._rules:
            rule(ctx)
            if ctx.cfg.collect_rule_masks:
                ctx.rule_masks[rule.__class__.__name__] = ctx.mask_valid.clone()

    def _finalise(self, ctx: CandidateContext) -> CandidateSamplingResult:
        mask_valid = ctx.mask_valid
        shell_poses = ctx.shell_poses
        assert shell_poses is not None

        assert shell_poses._data is not None
        poses_world_valid = PoseTW(shell_poses._data[mask_valid])
        reference_pose = ctx.reference_pose
        ref_inv = reference_pose.inverse()
        poses_ref_valid = ref_inv.compose(poses_world_valid)

        template = ctx.camera_calib_template
        if template is None:
            template_data = CameraTW.from_parameters()._data.to(poses_ref_valid._data.device).view(1, -1)
        else:
            template_data = template._data
            if template_data.ndim == 1:
                template_data = template_data.view(1, -1)
            template_data = template_data.to(poses_ref_valid._data.device)

        n = poses_ref_valid._data.shape[0]
        template_data = template_data[0].unsqueeze(0).expand(n, -1).clone()
        # Store camera pose in the reference frame as camera<-reference (EFM convention).
        poses_cam_ref = poses_ref_valid.inverse()
        template_data[:, 10:22] = poses_cam_ref._data

        poses_cam = CameraTW(template_data)

        return CandidateSamplingResult(
            views=poses_cam,
            reference_pose=reference_pose,
            mask_valid=mask_valid,
            masks=ctx.rule_masks if ctx.cfg.collect_rule_masks else {},
            shell_poses=shell_poses,
            shell_offsets_ref=ctx.shell_offsets_ref,
            extras=ctx.debug if ctx.cfg.collect_debug_stats else {},
        )


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

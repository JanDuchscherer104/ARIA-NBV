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

from collections.abc import Iterator
from contextlib import contextmanager
from math import radians
from typing import Annotated, Literal

import torch
import trimesh  # type: ignore[import-untyped]
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from pydantic import AliasChoices, Field, field_validator, model_validator

from ..data.efm_views import EfmSnippetView
from ..utils import BaseConfig, Console, Verbosity
from ..utils.frames import rotate_yaw_cw90, world_up_tensor
from .candidate_generation_rules import (
    FreeSpaceRule,
    MinDistanceToMeshRule,
    PathCollisionRule,
    Rule,
)
from .orientations import OrientationBuilder
from .positional_sampling import PositionSampler
from .types import (
    CandidateContext,
    CandidateSamplingResult,
    CollisionBackend,
    SamplingStrategy,
    ViewDirectionMode,
)


class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    """Configuration for sampling and pruning candidate camera poses around a reference frame.

    Encapsulates the radii/angle sampling envelope, orientation jitter options, collision and free-space
    filtering, and logging/debug controls used by :class:`CandidateViewGenerator`.
    """

    @property
    def target(self) -> type["CandidateViewGenerator"]:
        """Factory target for :meth:`BaseConfig.setup_target`."""
        return CandidateViewGenerator

    camera_label: Literal["rgb", "slaml", "slamr"] = "rgb"
    """Camera index to use for candidate generation."""

    num_samples: int = 60
    """Number of candidate poses requested after pruning."""
    oversample_factor: float = 2.0
    """Multiplicative oversampling factor applied before pruning to offset rejections."""
    max_resamples: int = 2
    """Maximum oversampling rounds if pruning removes too many candidates."""

    align_to_gravity: bool = True
    """If True, use a gravity-aligned copy of the reference pose for sampling.

    This removes pitch/roll from the sampling frame while keeping the reference yaw (forward direction projected
    onto the horizontal plane). It stabilises the sampling shell when the reference pose is strongly tilted
    (e.g., high roll angles).
    """

    min_radius: float = 0.5
    """Inner radius (metres) of the sampling shell around the reference pose."""
    max_radius: float = 1.8
    """Outer radius (metres) of the sampling shell around the reference pose."""

    min_elev_deg: float = -20.0
    """Minimum elevation angle (deg) relative to the world horizontal plane."""
    max_elev_deg: float = 25.0
    """Maximum elevation angle (deg) relative to the world horizontal plane."""
    delta_azimuth_deg: float = 170.0
    """Total azimuth spread (deg) around the reference forward direction; 360 unlocks full sphere."""

    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM_SPHERE
    """Distribution used to draw direction samples in the rig frame."""
    kappa: float = 4.0
    """Concentration parameter for the forward-biased PowerSpherical sampler."""

    min_distance_to_mesh: float = 0.2
    """Minimum clearance (metres) between candidate center and mesh surface."""
    ensure_collision_free: bool = True
    """Reject candidates whose straight path from the reference intersects the mesh."""
    ensure_free_space: bool = True
    """Constrain candidates to lie inside the snippet occupancy AABB."""
    collision_backend: CollisionBackend = CollisionBackend.P3D
    """Backend to use for collision and distance checks."""
    ray_subsample: int = 32
    """Number of samples per ray when using discretised collision checks."""
    step_clearance: float = 0.1
    """Distance threshold (metres) below which discretised collision samples are rejected."""

    mesh_samples: int | None = None
    """Optional number of mesh samples used by mesh-distance rules when applicable."""

    device: Annotated[torch.device, Field(default="auto")]
    """Torch device on which sampling and rule evaluation run (auto-select CUDA if available)."""
    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging.",
    )
    """Verbosity level for logging (0=quiet, 1=normal, 2=verbose)."""
    is_debug: bool = False
    """Enable debug logging and force verbose output when True."""

    collect_rule_masks: bool = False
    """Store per-rule boolean masks in the sampling result for diagnostics."""
    collect_debug_stats: bool = False
    """Allow rules to emit extra tensors (e.g., distances) in ``CandidateSamplingResult.extras``."""

    reference_frame_index: int | None = None
    """Optional camera frame index to use as reference pose; None defaults to the final pose."""

    # View orientation controls
    view_direction_mode: ViewDirectionMode = ViewDirectionMode.RADIAL_AWAY
    """Base orientation strategy for candidates."""

    view_sampling_strategy: SamplingStrategy | None = None
    """Optional view-direction sampler in the base camera frame (legacy path).

    Behaviour:
        - If either ``view_max_azimuth_deg`` or ``view_max_elevation_deg`` is > 0, view jitter is sampled as a
          bounded box in local yaw/pitch regardless of ``view_sampling_strategy``.
        - If both caps are 0, this field controls whether view directions are drawn from a distribution
          (PowerSpherical / uniform sphere) or kept deterministic (``None``).
    """

    view_kappa: float | None = None
    """Concentration for PowerSpherical view sampler; defaults to positional `kappa` when None."""

    view_max_angle_deg: float = 0.0
    """Fallback cap (deg) applied to both azimuth and elevation jitter when per-axis caps are unset."""

    view_max_azimuth_deg: float | None = 60.0
    """Maximum horizontal deviation (deg, +/-) from the base direction."""

    view_max_elevation_deg: float | None = 30.0
    """Maximum vertical deviation (deg, +/-) from the base direction."""

    view_roll_jitter_deg: float = 0.0
    """Symmetric roll jitter (deg) around the sampled forward axis in camera frame."""

    view_target_point_world: torch.Tensor | None = None
    """Optional world-space target for TARGET_POINT mode (shape (3,))."""

    seed: int | None = 0
    """Optional deterministic seed for candidate sampling.

    Set to ``None`` to keep the current global RNG state (non-deterministic).
    """

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, value: str | torch.device) -> torch.device:
        return super()._resolve_device(value)

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Verbosity | int | str) -> Verbosity:
        return Verbosity.from_any(value)

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, value: int | None) -> int | None:
        if value is not None and int(value) < 0:
            raise ValueError("seed must be >= 0 or None.")
        return value

    @field_validator(
        "view_max_angle_deg",
        "view_max_azimuth_deg",
        "view_max_elevation_deg",
        "view_roll_jitter_deg",
    )
    @classmethod
    def _non_negative_angles(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("Angular jitter caps must be non-negative.")
        return value

    @model_validator(mode="after")
    def set_debug(self) -> CandidateViewGeneratorConfig:
        if self.is_debug:
            object.__setattr__(self, "verbosity", Verbosity.VERBOSE)
        if self.view_kappa is None:
            object.__setattr__(self, "view_kappa", self.kappa)
        if self.view_max_azimuth_deg is None:
            object.__setattr__(self, "view_max_azimuth_deg", self.view_max_angle_deg)
        if self.view_max_elevation_deg is None:
            object.__setattr__(self, "view_max_elevation_deg", self.view_max_angle_deg)
        return self

    @property
    def min_elev_rad(self) -> float:
        return radians(self.min_elev_deg)

    @property
    def max_elev_rad(self) -> float:
        return radians(self.max_elev_deg)

    @property
    def delta_azimuth_rad(self) -> float:
        return radians(self.delta_azimuth_deg)


def _ensure_unbatched_pose(pose: PoseTW) -> PoseTW:
    """Ensure PoseTW has shape (12,) instead of (1,12) when singleton batch."""
    if pose._data.ndim == 2 and pose._data.shape[0] == 1:
        return PoseTW(pose._data.squeeze(0))
    return pose


def _gravity_align_pose(reference_pose: PoseTW, *, eps: float = 1e-6) -> PoseTW:
    """Return a gravity-aligned variant of ``reference_pose`` with identical translation.

    The aligned pose uses the VIO world-up axis (see :func:`oracle_rri.utils.frames.world_up_tensor`) and keeps
    the reference yaw by projecting the original forward axis onto the horizontal plane. This effectively removes
    pitch and roll so azimuth/elevation sampling caps behave as intended even when the reference camera is tilted.

    Args:
        reference_pose: ``PoseTW`` with shape ``(12,)`` or ``(1,12)`` (world<-reference).
        eps: Numerical stability guard for near-degenerate projections.

    Returns:
        ``PoseTW`` world<-reference pose with gravity-aligned rotation and unchanged translation.
    """
    reference_pose = _ensure_unbatched_pose(reference_pose)
    r_wr = reference_pose.R  # (..., 3, 3)
    t_w = reference_pose.t  # (..., 3)
    device = r_wr.device
    dtype = r_wr.dtype

    wup = world_up_tensor(device=device, dtype=dtype)  # (3,)

    fwd_w = r_wr[..., :, 2]  # (..., 3)
    fwd_h = fwd_w - (fwd_w * wup).sum(dim=-1, keepdim=True) * wup
    fwd_norm = fwd_h.norm(dim=-1, keepdim=True)

    left_w = r_wr[..., :, 0]  # (..., 3)
    left_h = left_w - (left_w * wup).sum(dim=-1, keepdim=True) * wup
    left_norm = left_h.norm(dim=-1, keepdim=True)

    # Expand world-up to match batch dimensions of the pose axes.
    wup_exp = wup
    while wup_exp.ndim < fwd_h.ndim:
        wup_exp = wup_exp.unsqueeze(0)
    wup_exp = wup_exp.expand_as(fwd_h)

    # Fallback when forward is near-parallel to gravity: derive forward from left×up.
    left_unit = left_h / left_norm.clamp_min(eps)
    fwd_from_left = torch.cross(left_unit, wup_exp, dim=-1)
    fwd_from_left = fwd_from_left / fwd_from_left.norm(dim=-1, keepdim=True).clamp_min(
        eps,
    )

    use_fallback = fwd_norm < eps
    fwd_unit = fwd_h / fwd_norm.clamp_min(eps)
    z_w = torch.where(use_fallback, fwd_from_left, fwd_unit)

    # Final fallback when both forward/left projections are degenerate.
    degenerate = z_w.norm(dim=-1, keepdim=True) < eps
    if degenerate.any():
        alt = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        alt = alt - (alt * wup).sum() * wup
        alt = alt / alt.norm().clamp_min(eps)
        alt_exp = alt
        while alt_exp.ndim < z_w.ndim:
            alt_exp = alt_exp.unsqueeze(0)
        alt_exp = alt_exp.expand_as(z_w)
        z_w = torch.where(degenerate, alt_exp, z_w)

    x_w = torch.cross(wup_exp, z_w, dim=-1)
    x_w = x_w / x_w.norm(dim=-1, keepdim=True).clamp_min(eps)
    y_w = torch.cross(z_w, x_w, dim=-1)

    r_new = torch.stack([x_w, y_w, z_w], dim=-1)
    return PoseTW.from_Rt(r_new, t_w)


@contextmanager
def _maybe_seed(seed: int | None, *, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return

    # IMPORTANT: `torch.random.fork_rng(devices=None)` will attempt to snapshot
    # CUDA RNG state and triggers CUDA initialization even when running on CPU.
    # Use an empty list unless we explicitly want to manage CUDA RNG state.
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cuda_devices = [int(idx)]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


class CandidateViewGenerator:
    """Generate candidate :class:`PoseTW` around a reference rig pose using composeable and modular rules.

    This class orchestrates the full candidate generation process:

    * positional sampling via :class:`PositionSampler`,
    * orientation construction via :class:`OrientationBuilder`, and
    * rule-based pruning via :class:`FreeSpaceRule`, :class:`MinDistanceToMeshRule` and :class:`PathCollisionRule`.

    """

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
        self,
        sample: EfmSnippetView,
        frame_index: int | None = None,
    ) -> CandidateSamplingResult:
        """Generate candidates using an :class:`EfmSnippetView` sample.

        Args:
            sample: Snippet view with trajectory and mesh.
            frame_index: Optional frame index to extract the reference pose instead of using the final pose.
                0 <= frame_index < F where is the number of frames in the snippet; F = sample.get_camera(self.config.camera_label).num_frames.
        """
        device = torch.device(self.config.device)
        occ = sample.get_occupancy_extend()
        self.console.dbg(
            f"Using occupancy extent: (xmin, xmax, ymin, ymax, zmin, zmax) = {occ}",
        )
        occupancy_extent = occ.to(device=device, dtype=torch.float32)
        gt_mesh = sample.mesh
        mesh_verts = sample.mesh_verts
        mesh_faces = sample.mesh_faces

        assert mesh_verts is not None and mesh_faces is not None, "Mesh vertices and faces must be provided."

        cam_view = sample.get_camera(self.config.camera_label)

        if frame_index is None:
            frame_index = self.config.reference_frame_index

        if frame_index is None:
            reference_pose = sample.trajectory.final_pose.to(device=device)
        else:
            cam_idx, traj_idx = cam_view.nearest_traj_indices(
                sample.trajectory.time_ns,
                [frame_index],
                default_last=True,
            )
            if traj_idx.numel() == 0:
                reference_pose = sample.trajectory.final_pose.to(device=device)
            else:
                reference_pose = sample.trajectory.t_world_rig[traj_idx].to(
                    device=device,
                )

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
        """Sample candidate poses around `reference_pose` and apply pruning rules.

        Samples candidate positions and orientations, wraps them in a :class:`CandidateContext`, runs all configured
        rules, and returns :class:`CandidateSamplingResult`.

        Args:
            reference_pose:
                World<-reference :class:`PoseTW` used as the physical rig pose. When
                ``align_to_gravity`` is enabled, a gravity-aligned copy of this pose
                defines the sampling frame (stored in ``CandidateSamplingResult.sampling_pose``).
            gt_mesh:
                Ground-truth :class:`trimesh.Trimesh` in the world frame for pruning.
            mesh_verts:
                `Tensor['V, 3']` mesh vertices aligned with :attr:`gt_mesh`.
            mesh_faces:
                `Tensor['F, 3']` integer vertex indices defining mesh faces.
            camera_calib_template:
                :class:`CameraTW` whose intrinsics/metadata are cloned for each candidate; its pose block is
                overwritten with candidate extrinsics.
            occupancy_extent:
                `Tensor['6']` world-space AABB used by :class:`FreeSpaceRule`.

        Returns:
            :class:`CandidateSamplingResult` holding the valid candidate :class:`CameraTW`, reference pose, shell
            poses, masks and optional debug statistics.
        """
        device = self.config.device

        reference_pose = rotate_yaw_cw90(
            _ensure_unbatched_pose(reference_pose.to(device)),
        )
        sampling_pose = _gravity_align_pose(reference_pose) if self.config.align_to_gravity else reference_pose

        with _maybe_seed(self.config.seed, device=torch.device(device)):
            centers_world, offsets_ref = PositionSampler(self.config).sample(
                sampling_pose,
            )
            shell_poses, view_dirs_delta = OrientationBuilder(self.config).build(
                sampling_pose,
                centers_world,
            )

        ctx = CandidateContext(
            cfg=self.config,
            reference_pose=reference_pose,
            sampling_pose=sampling_pose,
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts.to(device),
            mesh_faces=mesh_faces.to(device),
            occupancy_extent=occupancy_extent.to(device),
            camera_calib_template=camera_calib_template.to(device),
            shell_poses=shell_poses,
            centers_world=centers_world,
            shell_offsets_ref=offsets_ref,
            mask_valid=torch.ones(
                centers_world.shape[0],
                dtype=torch.bool,
                device=device,
            ),
            debug={"view_dirs_delta": view_dirs_delta} if view_dirs_delta is not None else {},
        )

        self._apply_rules(ctx)

        return self._finalise(ctx)

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
                ctx.record_mask(rule.__class__.__name__, ctx.mask_valid)

    def _finalise(self, ctx: CandidateContext) -> CandidateSamplingResult:
        mask_valid = ctx.mask_valid
        shell_poses = ctx.shell_poses
        assert shell_poses is not None
        assert shell_poses._data is not None

        poses_world_valid = PoseTW(shell_poses._data[mask_valid])  # world <- cam
        reference_pose = ctx.reference_pose  # world <- ref
        ref_inv = reference_pose.inverse()  # ref <- world
        poses_ref_valid = ref_inv.compose(poses_world_valid)  # ref <- cam

        template_data = _clone_camera_template(
            ctx.camera_calib_template,
            poses_ref_valid._data.shape[0],
            poses_ref_valid._data.device,
        )
        # Store camera pose in the reference frame as cam<-ref.
        poses_cam_ref = poses_ref_valid.inverse()
        template_data[:, CameraTW.T_CAM_RIG_IND] = poses_cam_ref._data

        poses_cam = CameraTW(template_data)

        return CandidateSamplingResult(
            views=poses_cam,
            reference_pose=reference_pose,
            sampling_pose=ctx.sampling_pose,
            mask_valid=mask_valid,
            masks=ctx.rule_masks if ctx.cfg.collect_rule_masks else {},
            shell_poses=shell_poses,
            shell_offsets_ref=ctx.shell_offsets_ref,
            extras=ctx.debug if ctx.cfg.collect_debug_stats else {},
        )


def _clone_camera_template(
    template: CameraTW,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast a camera template to `n` candidates on the target device."""
    data = template._data
    assert data is not None

    if data.ndim == 1:
        data = data.view(1, -1)
    return data.to(device)[0].unsqueeze(0).expand(n, -1).clone()


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "CollisionBackend",
    "SamplingStrategy",
]

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

from math import radians
from typing import Literal

import torch
import trimesh
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from pydantic import AliasChoices, Field, field_validator, model_validator

from ..data.efm_views import EfmSnippetView
from ..data.mesh_cache import mesh_from_snippet
from ..utils import BaseConfig, Console, Verbosity, select_device
from .candidate_generation_rules import FreeSpaceRule, MinDistanceToMeshRule, PathCollisionRule, Rule
from .orientations import OrientationBuilder
from .samplers import PositionSampler
from .types import (
    CandidateContext,
    CandidateSamplingResult,
    CollisionBackend,
    SamplingStrategy,
    ViewDirectionMode,
)


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
        Clearance enforced between camera center and mesh (metres).
    ensure_collision_free:
        Enable straight-line path collision filtering.
    ensure_free_space:
        Enable AABB workspace filtering.
    collision_backend:
        Backend for collision / distance checks (`P3D`, `PYEMBREE`, or
        `TRIMESH`).
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
        mesh) in `CandidateSamplingResult.extras`.
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
    """Concentration for PowerSpherical view sampler; defaults to positional `kappa` when None."""

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
        """Sample candidate poses around `reference_pose` and apply pruning rules.

        Samples candidate positions and orientations, wraps them in a :class:`CandidateContext`, runs all configured
        rules, and returns :class:`CandidateSamplingResult`.

        Args:
            reference_pose:
                reference2world :class:`PoseTW` that defines the sampling origin and local rig frame.
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
        cfg = self.config
        device = torch.device(cfg.device)

        with torch.no_grad():
            reference_pose = _ensure_unbatched_pose(reference_pose.to(device))
            sampler = PositionSampler(cfg)
            centers_world, offsets_ref = sampler.sample(reference_pose)
            shell_poses = OrientationBuilder(cfg).build(reference_pose, centers_world)

        ctx = CandidateContext(
            cfg=cfg,
            reference_pose=reference_pose,
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
        poses_world_valid = PoseTW(shell_poses._data[mask_valid])
        reference_pose = ctx.reference_pose
        ref_inv = reference_pose.inverse()
        poses_ref_valid = ref_inv.compose(poses_world_valid)

        template_data = _clone_camera_template(
            ctx.camera_calib_template, poses_ref_valid._data.shape[0], poses_ref_valid._data.device
        )
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


def _clone_camera_template(template: CameraTW, n: int, device: torch.device) -> torch.Tensor:
    """Broadcast a camera template to `n` candidates on the target device."""

    if template is None:
        data = CameraTW.from_parameters()._data
    else:
        data = template._data

    if data.ndim == 1:
        data = data.view(1, -1)
    return data.to(device)[0].unsqueeze(0).expand(n, -1).clone()


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

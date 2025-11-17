"""Candidate view generation with composable pruning rules.

Implements Config-as-Factory (see copilot-instructions). Sampling is vectorised
and GPU-capable; rules are modular and can be composed or replaced.

Frames: Aria/ATEK world (x-left, y-up, z-forward). Poses are `PoseTW` (T_A_B: maps
points from frame B into frame A).
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

import torch
import trimesh
from efm3d.aria import PoseTW
from pydantic import Field, field_validator, model_validator

from ..data.views import TypedSample
from ..utils import BaseConfig, Console
from .candidate_generation_rules import (
    FreeSpaceRule,
    MinDistanceToMeshRule,
    PathCollisionRule,
    Rule,
    ShellSamplingRule,
)
from .types import CandidateContext, CandidateSamplingResult, CollisionBackend, SamplingStrategy


# TODO we want to be able to display the space of potential candidate views as well as actually drawn candidate views!
class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    """Config for candidate generation.

    Attributes:
        target: factory target.
        num_samples: number of candidate poses to sample.
        max_resamples: maximum rounds of resampling to replace invalid candidates.
        min_radius/max_radius: spherical shell radii (m).
        min_elev_deg/max_elev_deg: elevation bounds for sampling (deg).
        azimuth_full_circle: if False sample half-sphere about forward axis.
        sampling_strategy: distribution for yaw/pitch sampling.
        ensure_collision_free: enable mesh-based path filtering.
        collision_backend: ray backend (pyembree if available, else trimesh).
        min_distance_to_mesh: clearance to enforce between pose and mesh (m).
        ensure_free_space: enable voxel-extent filtering.
        occupancy_extent: optional [6] tensor (xmin,xmax,ymin,ymax,zmin,zmax) in world.
        ray_subsample: number of samples along path for collision test.
        step_clearance: step size for distance checks (m).
        device: torch device for vectorised ops.
    """

    target: type["CandidateViewGenerator"] = Field(default_factory=lambda: CandidateViewGenerator, exclude=True)

    """Factory target for the config. Runtime class instantiated by `setup_target()`.
    This is excluded from serialization."""

    camera_index: Literal["rgb", "rgb_depth", "slam_l", "slam_r"] = "rgb_depth"
    """Camera index to use for candidate generation."""

    num_samples: int = 512
    """Number of candidate poses to sample per `generate` call."""
    max_resamples: int = 3
    """Maximum rounds of resampling to replace invalid candidates."""
    min_radius: float = 0.4
    """Minimum radius (m) for spherical-shell sampling around `last_pose`."""
    max_radius: float = 1.6
    """Maximum radius (m) for spherical-shell sampling around `last_pose`."""
    min_elev_deg: float = -15.0
    """Minimum elevation angle (degrees) for sampled view directions."""
    max_elev_deg: float = 45.0
    """Maximum elevation angle (degrees) for sampled view directions."""
    azimuth_full_circle: bool = True
    """If True sample full 360deg azimuth; otherwise sample a half-sphere about forward axis."""
    sampling_strategy: SamplingStrategy = SamplingStrategy.SHELL_UNIFORM
    """Distribution strategy for yaw/pitch sampling (area-uniform or forward-biased)."""

    ensure_collision_free: bool = True
    """Enable straight-line path collision filtering.

    When True, `_rule_path_collision` casts a ray from the last camera pose
    to each candidate viewpoint and rejects candidates whose path intersects
    the GT mehs.
    """

    collision_backend: CollisionBackend = CollisionBackend.PYEMBREE
    """Backend for ray/mesh intersection tests.

    `PYEMBREE` uses `RayMeshIntersector` (if installed) for fast queries;
    `TRIMESH` falls back to the pure-Python trimesh ray engine.
    """

    min_distance_to_mesh: float = 0.05
    """Minimum clearance (metres) between the candidate camera centre and
    the GT mesh.

    `_rule_min_distance_to_mesh` rejects candidates whose camera centre is
    closer than this threshold, preventing views that start "inside" walls
    or geometry.
    """

    ensure_free_space: bool = True
    """Enable workspace AABB filtering.

    When True, `_rule_free_space` enforces that candidate camera centres
    lie inside an axis-aligned bounding box defined by `occupancy_extent`.
    Use this to restrict sampling to the room / region covered by the
    snippet's points or mesh.
    """

    occupancy_extent: torch.Tensor | None = None  # [6]
    """Optional [6] tensor (xmin,xmax,ymin,ymax,zmin,zmax) describing allowed workspace bounds."""

    ray_subsample: int = 128
    """Number of ray samples / subdivisions when evaluating path collisions."""
    step_clearance: float = 0.05
    """Step size (m) for distance checks along paths when required."""
    device: Annotated[torch.device, Literal["cuda", "cpu"]] = "cuda"  # type: ignore[assignment]
    """Preferred torch device for vectorised operations (e.g., 'cuda' or 'cpu')."""

    verbose: bool = True

    is_debug: bool = False
    """If True, enable debug logging and set device to 'cpu'."""

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, v: str) -> torch.device:
        value = v.lower()
        if value.startswith("cuda") and torch.cuda.is_available():
            return torch.device(value)
        return torch.device("cpu") if value.startswith("cuda") else torch.device(value)

    @model_validator(mode="after")
    def set_debug(self) -> Self:
        if self.is_debug:
            object.__setattr__(self, "device", torch.device("cpu"))
            object.__setattr__(self, "verbose", True)
            Console.with_prefix(self.__class__.__name__, "set_debug").set_verbose(True).log(
                "Debug mode enabled: forcing device to 'cpu' and verbose logging."
            )
        return self


class CandidateViewGenerator:
    """Generate candidate `PoseTW` around latest pose with composable rules."""

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(True)
        self.rules: list[Rule] = []
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        self.rules = [ShellSamplingRule(self.config), MinDistanceToMeshRule(self.config)]
        if self.config.ensure_collision_free:
            self.rules.append(PathCollisionRule(self.config))
        if self.config.ensure_free_space:
            self.rules.append(FreeSpaceRule(self.config))

    def generate_from_typed_sample(self, sample: TypedSample) -> CandidateSamplingResult:
        """Generate candidate poses using data from a TypedSample.

        Args:
            sample: TypedSample containing last pose and optional GT mesh/occupancy.
        Returns:
            CandidateSamplingResult with sampled poses and masks.
        """
        occupancy_extent = self._occupancy_extent_from_sample(sample)
        gt_mesh = sample.mesh
        last_pose = sample.trajectory.final_pose

        return self.generate(
            last_pose=last_pose,
            gt_mesh=gt_mesh,
            occupancy_extent=occupancy_extent,
        )

    def _occupancy_extent_from_sample(
        self,
        sample: TypedSample,
    ) -> torch.Tensor | None:
        """Return [xmin,xmax,ymin,ymax,zmin,zmax] AABB in world frame."""
        if sample.has_mesh:
            bounds = torch.from_numpy(sample.gt_mesh.bounds).to(device=self.config.device, dtype=torch.float32)
            vmin, vmax = bounds[0], bounds[1]
            return torch.stack(
                [vmin[0], vmax[0], vmin[1], vmax[1], vmin[2], vmax[2]],
                dim=0,
            )
        sem = sample.semidense
        if sem.volume_min is not None and sem.volume_max is not None:
            vmin = sem.volume_min.to(device=self.config.device, dtype=torch.float32)
            vmax = sem.volume_max.to(device=self.config.device, dtype=torch.float32)
            return torch.stack(
                [vmin[0], vmax[0], vmin[1], vmax[1], vmin[2], vmax[2]],
                dim=0,
            )
        return None

    # TODO: sampling must be done from the pose belonging to the camera_index specified in the config and also consider the cameras intrinsics to do correct sampling!
    def generate(
        self,
        last_pose: PoseTW,
        gt_mesh: trimesh.Trimesh | None = None,
        occupancy_extent: torch.Tensor | None = None,
    ) -> CandidateSamplingResult:
        """Sample candidate poses and apply pruning rules.

        Args:
            last_pose: current rig pose (PoseTW).
            gt_mesh: optional mesh for collision checks.
            occupancy_extent: optional [6] bounds (overrides config.occupancy_extent).

        Returns:
            dict with:
                poses: PoseTW of valid candidates.
                mask_valid: bool mask before wrapping to PoseTW.
                masks: list of per-rule masks.
        """

        device = self.config.device
        ctx: CandidateContext = {
            "last_pose": last_pose.to(device),
            "gt_mesh": gt_mesh,
            "occupancy_extent": occupancy_extent if occupancy_extent is not None else self.config.occupancy_extent,
            "device": device,
            "poses": torch.empty(0, 12, device=device),
            "mask": torch.empty(0, dtype=torch.bool, device=device),
        }
        poses_accum: list[torch.Tensor] = []
        masks_accum: list[torch.Tensor] = []
        shell_accum: list[PoseTW] = []
        remaining = self.config.num_samples
        attempts = 0

        while remaining > 0 and attempts < self.config.max_resamples:
            ctx_batch = self._seed(ctx, remaining)
            masks: list[torch.Tensor] = []
            for rule in self.rules:
                ctx_batch = rule(ctx_batch)
                masks.append(ctx_batch["mask"])
            # preserve the raw sampled poses (before masking) for visualization of the full candidate space
            shell_accum.append(ctx_batch["poses"].clone())
            mask_valid = (
                torch.stack(masks, dim=0).all(dim=0)
                if masks
                else torch.ones(ctx_batch["poses"].shape[0], dtype=torch.bool, device=device)
            )
            if mask_valid.any():
                poses_accum.append(ctx_batch["poses"][mask_valid])
            masks_accum.extend(masks)
            remaining = self.config.num_samples - sum(p.shape[0] for p in poses_accum)
            attempts += 1

        if poses_accum:
            poses_cat = torch.cat(poses_accum, dim=0)[: self.config.num_samples]
            mask_valid_out = torch.ones(poses_cat.shape[0], dtype=torch.bool, device=device)
        else:
            poses_cat = torch.zeros(self.config.num_samples, 12, device=device)
            mask_valid_out = torch.zeros(self.config.num_samples, dtype=torch.bool, device=device)
        poses_valid = PoseTW(poses_cat)
        shell_poses = (
            torch.cat([p._data for p in shell_accum], dim=0)
            if shell_accum
            else torch.zeros(0, 12, device=device)
        )
        return {
            "poses": poses_valid,
            "mask_valid": mask_valid_out,
            "masks": masks_accum,
            "shell_poses": shell_poses,
        }

    # ------------------------------------------------------------------ rules
    def _seed(self, ctx: CandidateContext, n: int) -> CandidateContext:
        """Initialise pose tensor placeholder."""
        ctx["poses"] = torch.zeros(n, 12, device=ctx["device"])
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=ctx["device"])
        return ctx


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

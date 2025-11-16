"""Candidate view generation with composable pruning rules.

Implements Config-as-Factory (see copilot-instructions). Sampling is vectorised
and GPU-capable; rules are modular and can be composed or replaced.

Frames: Aria/ATEK world (x-left, y-up, z-forward). Poses are `PoseTW` (T_A_B: maps
points from frame B into frame A).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
import trimesh
from efm3d.aria import PoseTW
from pydantic import Field

from ..utils import BaseConfig, Console

Rule = Callable[[torch.Tensor, dict], dict]  # rule(poses, ctx) -> ctx with updated mask


class SamplingStrategy(str, Enum):
    """Sampling strategy for candidate viewpoints."""

    SHELL_UNIFORM = "shell_uniform"
    FORWARD_GAUSSIAN = "forward_gaussian"


class CollisionBackend(str, Enum):
    """Backend for collision tests."""

    PYEMBREE = "pyembree"
    TRIMESH = "trimesh"


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

    num_samples: int = 512
    max_resamples: int = 3
    min_radius: float = 0.4
    max_radius: float = 1.6
    min_elev_deg: float = -15.0
    max_elev_deg: float = 45.0
    azimuth_full_circle: bool = True
    sampling_strategy: SamplingStrategy = SamplingStrategy.SHELL_UNIFORM

    ensure_collision_free: bool = True
    collision_backend: CollisionBackend = CollisionBackend.PYEMBREE
    min_distance_to_mesh: float = 0.05
    ensure_free_space: bool = True
    occupancy_extent: torch.Tensor | None = None  # [6]

    ray_subsample: int = 128
    step_clearance: float = 0.05
    device: str = "cuda"


class CandidateViewGenerator:
    """Generate candidate `PoseTW` around latest pose with composable rules."""

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(True)
        self.rules: list[Rule] = []
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        cfg = self.config
        self.rules = [self._rule_shell_sampling, self._rule_min_distance_to_mesh]
        if cfg.ensure_collision_free:
            self.rules.append(self._rule_path_collision)
        if cfg.ensure_free_space:
            self.rules.append(self._rule_free_space)

    def generate(
        self,
        last_pose: PoseTW,
        gt_mesh: trimesh.Trimesh | None = None,
        occupancy_extent: torch.Tensor | None = None,
    ) -> dict:
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

        device_name = self.config.device if (self.config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        device = torch.device(device_name)
        ctx = {
            "last_pose": last_pose.to(device),
            "gt_mesh": gt_mesh,
            "occupancy_extent": occupancy_extent if occupancy_extent is not None else self.config.occupancy_extent,
            "device": device,
        }
        poses_accum: list[torch.Tensor] = []
        masks_accum: list[torch.Tensor] = []
        remaining = self.config.num_samples
        attempts = 0

        while remaining > 0 and attempts < self.config.max_resamples:
            ctx_batch = self._seed(ctx, remaining)
            masks: list[torch.Tensor] = []
            for rule in self.rules:
                ctx_batch = rule(ctx_batch["poses"], ctx_batch)
                masks.append(ctx_batch["mask"])
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
        return {"poses": poses_valid, "mask_valid": mask_valid_out, "masks": masks_accum}

    # ------------------------------------------------------------------ rules
    def _seed(self, ctx: dict, n: int) -> dict:
        """Initialise pose tensor placeholder."""
        ctx["poses"] = torch.zeros(n, 12, device=ctx["device"])
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=ctx["device"])
        return ctx

    def _rule_shell_sampling(self, ctx: dict) -> dict:
        cfg = self.config
        n = ctx["poses"].shape[0]
        dev = ctx["device"]
        r = torch.rand(n, device=dev) * (cfg.max_radius - cfg.min_radius) + cfg.min_radius
        az = torch.rand(n, device=dev) * (2 * torch.pi if cfg.azimuth_full_circle else torch.pi)
        pos_local = self._sample_directions(n, az, dev, cfg)
        pos_local = pos_local * r.unsqueeze(1)

        pose_last = ctx["last_pose"].to_matrix()
        r_mat = pose_last[:3, :3]
        t = pose_last[:3, 3]
        pos_world = (r_mat @ pos_local.T).T + t

        forward = torch.nn.functional.normalize(t - pos_world, dim=1)
        up = torch.tensor([0, 1, 0], device=dev).expand_as(forward)
        right = torch.nn.functional.normalize(torch.cross(forward, up, dim=1), dim=1)
        up_corrected = torch.cross(right, forward, dim=1)
        r_cam = torch.stack([right, up_corrected, forward], dim=2)  # [n,3,3]
        rt = torch.cat([r_cam, pos_world.unsqueeze(2)], dim=2).reshape(n, 12)
        ctx["poses"] = rt
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=dev)
        return ctx

    def _sample_directions(
        self, n: int, az: torch.Tensor, device: torch.device, cfg: CandidateViewGeneratorConfig
    ) -> torch.Tensor:
        """Sample direction unit vectors given yaw samples and config elevation band."""

        min_elev = torch.deg2rad(torch.tensor(cfg.min_elev_deg, device=device))
        max_elev = torch.deg2rad(torch.tensor(cfg.max_elev_deg, device=device))

        if cfg.sampling_strategy == SamplingStrategy.FORWARD_GAUSSIAN:
            mean = (min_elev + max_elev) / 2
            std = (max_elev - min_elev) / 4
            elev = torch.clamp(torch.randn(n, device=device) * std + mean, min=min_elev, max=max_elev)
        else:
            # area-uniform over spherical band using inverse CDF of sin(elev)
            sin_min, sin_max = torch.sin(min_elev), torch.sin(max_elev)
            u = torch.rand(n, device=device) * (sin_max - sin_min) + sin_min
            elev = torch.arcsin(torch.clamp(u, -0.999999, 0.999999))

        x = torch.cos(elev) * torch.cos(az)
        y = torch.sin(elev)
        z = torch.cos(elev) * torch.sin(az)
        return torch.stack([x, y, z], dim=1)

    def _rule_min_distance_to_mesh(self, poses: torch.Tensor, ctx: dict) -> dict:
        """Disallow viewpoints closer than `min_distance_to_mesh` to the GT mesh."""

        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None or self.config.min_distance_to_mesh <= 0:
            ctx["mask"] = mask
            return ctx

        positions = poses[:, 9:12]
        query = trimesh.proximity.ProximityQuery(mesh)
        dist = query.signed_distance(positions.detach().cpu().numpy())
        clear = torch.from_numpy(dist).to(mask.device) > self.config.min_distance_to_mesh
        ctx["mask"] = mask & clear
        return ctx

    def _rule_path_collision(self, poses: torch.Tensor, ctx: dict) -> dict:
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None:
            ctx["mask"] = mask
            return ctx
        pose_last = ctx["last_pose"].to_matrix()
        origin = pose_last[:3, 3].view(1, 3)
        targets = poses[:, 9:12]
        dirs = targets - origin
        dists = torch.linalg.norm(dirs, dim=1).clamp(min=1e-6)
        dirs_norm = dirs / dists.unsqueeze(1)

        origins_np = origin.expand_as(targets).detach().cpu().numpy()
        dirs_np = dirs_norm.detach().cpu().numpy()
        max_dist = dists.detach().cpu().numpy()
        ray_engine = mesh.ray
        if self.config.collision_backend == CollisionBackend.PYEMBREE:
            try:
                from trimesh.ray.ray_pyembree import RayMeshIntersector

                ray_engine = RayMeshIntersector(mesh)
            except ImportError:
                pass

        intersects = ray_engine.intersects_any(
            origins_np,
            dirs_np,
            multiple_hits=False,
            max_distance=max_dist,
        )
        free = torch.from_numpy(~intersects).to(mask.device)
        ctx["mask"] = mask & free
        return ctx

    def _rule_free_space(self, poses: torch.Tensor, ctx: dict) -> dict:
        mask = ctx["mask"]
        extent = ctx.get("occupancy_extent")
        if extent is None:
            ctx["mask"] = mask
            return ctx
        extent = extent.to(mask.device)
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        p = poses[:, 9:12]
        in_box = (
            (p[:, 0] >= xmin)
            & (p[:, 0] <= xmax)
            & (p[:, 1] >= ymin)
            & (p[:, 1] <= ymax)
            & (p[:, 2] >= zmin)
            & (p[:, 2] <= zmax)
        )
        ctx["mask"] = mask & in_box
        return ctx


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

"""Rule implementations for candidate view generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
import trimesh
from efm3d.aria import PoseTW

from oracle_rri.utils.frames import view_axes_from_points

from .types import CandidateContext, CollisionBackend, SamplingStrategy

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig


class Rule(Protocol):
    def __call__(self, ctx: CandidateContext) -> CandidateContext: ...


class ShellSamplingRule:
    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        cfg = self.config
        n = ctx["poses"].shape[0]
        dev = ctx["device"]
        r = torch.rand(n, device=dev) * (cfg.max_radius - cfg.min_radius) + cfg.min_radius
        az = torch.rand(n, device=dev) * (2 * torch.pi if cfg.azimuth_full_circle else torch.pi)
        pos_local = self._sample_directions(n, az, dev)
        pos_local = pos_local * r.unsqueeze(1)

        pose_last: PoseTW = ctx["last_pose"]
        pos_world = pose_last.transform(pos_local)

        # Orient each candidate camera to look back at the last pose
        look_at = pose_last.t.expand_as(pos_world)
        r_cam = view_axes_from_points(cam_pos=pos_world, look_at=look_at)
        poses_tw = PoseTW.from_Rt(r_cam, pos_world)
        ctx["poses"] = poses_tw
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=dev)
        return ctx

    def _sample_directions(self, n: int, az: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_elev = torch.deg2rad(torch.tensor(self.config.min_elev_deg, device=device))
        max_elev = torch.deg2rad(torch.tensor(self.config.max_elev_deg, device=device))

        match self.config.sampling_strategy:
            case SamplingStrategy.FORWARD_GAUSSIAN:
                mean = (min_elev + max_elev) / 2
                std = (max_elev - min_elev) / 4
                elev = torch.clamp(torch.randn(n, device=device) * std + mean, min=min_elev, max=max_elev)
            case SamplingStrategy.SHELL_UNIFORM:
                sin_min, sin_max = torch.sin(min_elev), torch.sin(max_elev)
                u = torch.rand(n, device=device) * (sin_max - sin_min) + sin_min
                elev = torch.arcsin(torch.clamp(u, -0.999999, 0.999999))

        x = torch.cos(elev) * torch.cos(az)
        y = torch.sin(elev)
        z = torch.cos(elev) * torch.sin(az)
        return torch.stack([x, y, z], dim=1)


class MinDistanceToMeshRule:
    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None or self.config.min_distance_to_mesh <= 0:
            ctx["mask"] = mask
            return ctx

        positions = ctx["poses"].t  # PoseTW.t -> (N,3)
        query = trimesh.proximity.ProximityQuery(mesh)
        dist = query.signed_distance(positions.detach().cpu().numpy())
        clear = torch.from_numpy(dist).to(mask.device) > self.config.min_distance_to_mesh
        ctx["mask"] = mask & clear
        return ctx


class PathCollisionRule:
    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None:
            ctx["mask"] = mask
            return ctx
        origin = ctx["last_pose"].t.view(1, 3)
        poses = ctx["poses"]
        targets = poses.t  # (N,3)
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


class FreeSpaceRule:
    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        mask = ctx["mask"]
        extent = ctx.get("occupancy_extent")
        if extent is None:
            ctx["mask"] = mask
            return ctx
        extent = extent.to(mask.device)
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        p = ctx["poses"].t
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


def _sample_directions(
    n: int, az: torch.Tensor, device: torch.device, cfg: CandidateViewGeneratorConfig
) -> torch.Tensor:
    min_elev = torch.deg2rad(torch.tensor(cfg.min_elev_deg, device=device))
    max_elev = torch.deg2rad(torch.tensor(cfg.max_elev_deg, device=device))

    if cfg.sampling_strategy == SamplingStrategy.FORWARD_GAUSSIAN:
        mean = (min_elev + max_elev) / 2
        std = (max_elev - min_elev) / 4
        elev = torch.clamp(torch.randn(n, device=device) * std + mean, min=min_elev, max=max_elev)
    else:
        sin_min, sin_max = torch.sin(min_elev), torch.sin(max_elev)
        u = torch.rand(n, device=device) * (sin_max - sin_min) + sin_min
        elev = torch.arcsin(torch.clamp(u, -0.999999, 0.999999))

    x = torch.cos(elev) * torch.cos(az)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.sin(az)
    return torch.stack([x, y, z], dim=1)


__all__ = [
    "Rule",
    "ShellSamplingRule",
    "MinDistanceToMeshRule",
    "PathCollisionRule",
    "FreeSpaceRule",
]

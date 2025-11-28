"""Pruning rules for candidate pose generation."""

from __future__ import annotations

from typing import Protocol

import torch
import trimesh  # type: ignore[import-untyped]
from git import TYPE_CHECKING

from ..utils import Console
from .types import CandidateContext, CollisionBackend

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig


class Rule(Protocol):
    """Callable pruning rule."""

    def __call__(self, ctx: CandidateContext) -> None: ...


class MinDistanceToMeshRule:
    """Reject candidates whose centres are closer than a threshold to the mesh."""

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> None:
        if ctx.gt_mesh is None or ctx.centers_world is None or ctx.mask_valid is None:
            return

        need_distance = self.config.min_distance_to_mesh > 0 or ctx.cfg.collect_debug_stats
        if not need_distance:
            return

        positions = ctx.centers_world
        backend = self.config.collision_backend

        if backend == CollisionBackend.P3D and ctx.mesh_verts is not None and ctx.mesh_faces is not None:
            dist_t = point_mesh_distance(positions, ctx.mesh_verts, ctx.mesh_faces)
        else:
            if backend == CollisionBackend.P3D and (ctx.mesh_verts is None or ctx.mesh_faces is None):
                Console.with_prefix(self.__class__.__name__).warn(
                    "P3D backend selected for MinDistanceToMeshRule but mesh vertices/faces are missing; "
                    "falling back to trimesh proximity."
                )
            try:
                query = trimesh.proximity.ProximityQuery(ctx.gt_mesh)
                dist_np = query.signed_distance(positions.detach().cpu().numpy())
                dist_t = torch.from_numpy(dist_np).to(positions.device, positions.dtype).abs()
            except ModuleNotFoundError:
                verts = torch.as_tensor(ctx.gt_mesh.vertices, device=positions.device, dtype=torch.float32)
                faces = torch.as_tensor(ctx.gt_mesh.faces, device=positions.device, dtype=torch.int64)
                dist_t = point_mesh_distance(positions, verts, faces)

        if ctx.cfg.collect_debug_stats:
            ctx.debug["min_distance_to_mesh"] = dist_t

        if self.config.min_distance_to_mesh > 0:
            keep = dist_t > self.config.min_distance_to_mesh
            ctx.mask_valid = ctx.mask_valid & keep


class PathCollisionRule:
    """Reject candidates whose straight-line path from the last pose hits the mesh."""

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> None:
        if ctx.gt_mesh is None or ctx.centers_world is None or ctx.mask_valid is None:
            return

        origin = ctx.last_pose.t.view(1, 3)
        targets = ctx.centers_world
        dirs = targets - origin
        dists = dirs.norm(dim=1).clamp_min(1e-6)
        dirs_norm = dirs / dists.unsqueeze(1)

        backend = self.config.collision_backend

        if backend == CollisionBackend.P3D and ctx.mesh_verts is not None and ctx.mesh_faces is not None:
            steps = max(2, int(self.config.ray_subsample))
            t_vals = torch.linspace(0.0, 1.0, steps, device=targets.device, dtype=targets.dtype)
            pts = origin.view(1, 1, 3) + dirs_norm.unsqueeze(1) * (t_vals.view(1, -1, 1) * dists.view(-1, 1, 1))
            pts_flat = pts.reshape(-1, 3)
            dists_pts = point_mesh_distance(pts_flat, ctx.mesh_verts, ctx.mesh_faces).view(targets.shape[0], steps)
            collide = (dists_pts < self.config.step_clearance).any(dim=1)
            ctx.mask_valid = ctx.mask_valid & (~collide)
            return

        origins_np = origin.expand_as(targets).detach().cpu().numpy()
        dirs_np = dirs_norm.detach().cpu().numpy()
        max_dist = dists.detach().cpu().numpy()
        ray_engine = ctx.gt_mesh.ray
        if backend == CollisionBackend.PYEMBREE:
            try:
                from trimesh.ray.ray_pyembree import RayMeshIntersector

                ray_engine = RayMeshIntersector(ctx.gt_mesh)
            except ImportError:
                pass

        intersects = ray_engine.intersects_any(origins_np, dirs_np, multiple_hits=False, max_distance=max_dist)
        free = torch.from_numpy(~intersects).to(ctx.mask_valid.device)
        ctx.mask_valid = ctx.mask_valid & free


class FreeSpaceRule:
    """Restrict candidate centres to a world-space AABB."""

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> None:
        if ctx.occupancy_extent is None or ctx.centers_world is None or ctx.mask_valid is None:
            return

        extent = ctx.occupancy_extent.to(ctx.centers_world.device)
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        p = ctx.centers_world
        in_box = (
            (p[:, 0] >= xmin)
            & (p[:, 0] <= xmax)
            & (p[:, 1] >= ymin)
            & (p[:, 1] <= ymax)
            & (p[:, 2] >= zmin)
            & (p[:, 2] <= zmax)
        )
        ctx.mask_valid = ctx.mask_valid & in_box


def point_mesh_distance(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # TODO: Move this to a new utils module!
    """Compute point-to-mesh distances using PyTorch3D."""

    from pytorch3d.loss.point_mesh_distance import (  # type: ignore[import-untyped]
        _DEFAULT_MIN_TRIANGLE_AREA,
        point_face_distance,
    )

    device = points.device
    points = points.to(device)
    verts = verts.to(device)
    faces = faces.to(device)

    tris = verts[faces]
    points_first_idx = torch.zeros(1, device=device, dtype=torch.int64)
    tris_first_idx = torch.zeros(1, device=device, dtype=torch.int64)
    max_points = points.shape[0]

    dist_sq = point_face_distance(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        _DEFAULT_MIN_TRIANGLE_AREA,
    )
    return torch.sqrt(dist_sq)


__all__ = [
    "Rule",
    "MinDistanceToMeshRule",
    "PathCollisionRule",
    "FreeSpaceRule",
]

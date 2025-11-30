"""Pruning rules for candidate pose generation."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Protocol

import torch
import trimesh  # type: ignore[import-untyped]

from ..utils import Console
from .geometry import point_mesh_distance
from .types import CandidateContext, CollisionBackend

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig


class Rule(Protocol):
    """Callable pruning rule."""

    def __call__(self, ctx: CandidateContext) -> None: ...


class RuleBase:
    """Shared utilities for pruning rules (logging and mask helpers)."""

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._warned_backend = False

    def warn_once(self, message: str) -> None:
        if not self._warned_backend:
            self.console.warn(message)
            self._warned_backend = True


class MinDistanceToMeshRule(RuleBase):
    r"""Reject candidates whose centers are too close to the GT mesh.

    For each candidate center :math:`c_i` and mesh :math:`\mathcal{M}`, this rule computes the distance

    .. math::

        d_i = \min_{x \in \mathcal{M}} \lVert c_i - x \rVert_2

    and rejects candidates with :math:`d_i \leq \text{min_distance_to_mesh}`.

    When `cfg.collect_debug_stats` is True, the per-candidate distances are stored as
    `ctx.debug['min_distance_to_mesh']` for later analysis.
    """

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        super().__init__(config)

    def __call__(self, ctx: CandidateContext) -> None:
        """Mark candidates closer than threshold to the mesh as invalid.

        Updates `ctx.mask_valid` in place and records `min_distance_to_mesh` in `ctx.debug`
        when `collect_debug_stats` is enabled.
        """
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
                self.warn_once(
                    "P3D backend selected but mesh vertices/faces are missing; falling back to trimesh proximity."
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
            ctx.mark_debug("min_distance_to_mesh", dist_t)

        if self.config.min_distance_to_mesh > 0:
            keep = dist_t > self.config.min_distance_to_mesh
            ctx.mask_valid = ctx.mask_valid & keep


class PathCollisionRule(RuleBase):
    """Reject candidates whose straight-line path from reference hits the mesh.

    This rule enforces that the straight segment from the reference rig position to each candidate center does not
    intersect the mesh, optionally with a configurable clearance.

    Depending on :class:`CollisionBackend`, collision checks are implemented either by discretised distance sampling
    (PyTorch3D) or by analytic ray-mesh intersection tests (Trimesh / PyEmbree).

    The method:

    1. Returns early if no mesh is available or the step clearance is non-positive.
    2. Constructs a ray from the reference position to each candidate center.
    3. Depending on :attr:`config.collision_backend`:

        * :data:`CollisionBackend.P3D`:
            discretise each segment into ``ray_subsample`` points, compute distances via :func:`point_mesh_distance`,
            and mark collisions where any sample falls below ``step_clearance``.
        * :data:`CollisionBackend.TRIMESH` / :data:`CollisionBackend.PYEMBREE`:
            cast rays with maximum distance equal to the segment length and use the ray engine's
            :meth:`intersects_any` to identify collisions.

    4. Records the boolean collision mask in ``ctx.debug['path_collision_mask']`` when debug stats are enabled.
    5. Calls :meth:`CandidateContext.invalidate` to apply the collision mask as a rejection mask.
    """

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        super().__init__(config)
        self._pyembree_available = False
        if self.config.collision_backend == CollisionBackend.PYEMBREE:
            self._pyembree_available = importlib.util.find_spec("trimesh.ray.ray_pyembree") is not None

    def __call__(self, ctx: CandidateContext) -> None:
        """Reject candidates whose straight-line path from reference to center intersects the mesh."""
        if ctx.gt_mesh is None or ctx.centers_world is None or ctx.mask_valid is None:
            return

        if self.config.step_clearance <= 0:
            return

        origin = ctx.reference_pose.t.view(1, 3)
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
            if ctx.cfg.collect_debug_stats:
                ctx.mark_debug("path_collision_mask", collide)
            ctx.invalidate(collide)
            return

        origins_np = origin.expand_as(targets).detach().cpu().numpy()
        dirs_np = dirs_norm.detach().cpu().numpy()
        max_dist = dists.detach().cpu().numpy()
        ray_engine = ctx.gt_mesh.ray
        if backend == CollisionBackend.PYEMBREE and self._pyembree_available:
            from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore

            ray_engine = RayMeshIntersector(ctx.gt_mesh)
        elif backend == CollisionBackend.PYEMBREE and not self._pyembree_available:
            self.warn_once("pyembree not available; falling back to trimesh ray engine.")

        intersects = ray_engine.intersects_any(origins_np, dirs_np, multiple_hits=False, max_distance=max_dist)
        collide = torch.from_numpy(intersects).to(ctx.mask_valid.device)
        if ctx.cfg.collect_debug_stats:
            ctx.mark_debug("path_collision_mask", collide)
        ctx.invalidate(collide)


class FreeSpaceRule(RuleBase):
    """Restrict candidate centers to a world-space AABB."""

    def __init__(self, config: "CandidateViewGeneratorConfig"):
        super().__init__(config)

    def __call__(self, ctx: CandidateContext) -> None:
        """Keep only candidates inside the configured occupancy extent (AABB)."""
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


__all__ = ["Rule", "RuleBase", "MinDistanceToMeshRule", "PathCollisionRule", "FreeSpaceRule"]

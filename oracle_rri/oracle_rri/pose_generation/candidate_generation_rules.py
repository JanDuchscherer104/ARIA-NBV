"""Rule implementations for candidate view generation.

This module contains *local* rules that operate on a shared
[:class:`oracle_rri.pose_generation.types.CandidateContext`] and either
populate candidate poses or prune them:

* :class:`ShellSamplingRule` - draws candidate camera centres on a spherical
  shell around the last pose and orients them to look back at the trajectory.
* :class:`MinDistanceToMeshRule` - enforces a minimum Euclidean distance
  between the camera centre and the GT mesh using
  [:class:`trimesh.proximity.ProximityQuery`](https://trimesh.org/trimesh.proximity.html#trimesh.proximity.ProximityQuery.signed_distance).
* :class:`PathCollisionRule` - rejects candidates whose straight-line path
  from the last pose intersects the mesh, using either the trimesh ray
  interface or
  [:class:`trimesh.ray.ray_pyembree.RayMeshIntersector`](https://trimesh.org/trimesh.ray.ray_pyembree.html#trimesh.ray.ray_pyembree.RayMeshIntersector).
* :class:`FreeSpaceRule` - clips candidates to a world-space AABB derived
  from occupancy statistics.

All rules assume the **VIO world frame** used by EFM3D: gravity points along
``[0, 0, -g]`` and camera frames are RDF (X-right, Y-down, Z-forward).
"""

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
    """Callable interface for a candidate-generation rule."""

    def __call__(self, ctx: CandidateContext) -> CandidateContext: ...


class ShellSamplingRule:
    """Sample candidate poses on a spherical shell around the last pose.

    The rule operates in three stages:

    1. Sample radii :math:`r \\sim \\mathcal{U}(r_{\\min}, r_{\\max})` and
       directions on a spherical cap defined by
       ``min_elev_deg`` / ``max_elev_deg`` and ``azimuth_full_circle``.
       Directions are parameterised as

       .. math::

           x &= \\cos(\\text{elev}) \\cos(\\text{az}), \\\\
           y &= \\sin(\\text{elev}), \\\\
           z &= \\cos(\\text{elev}) \\sin(\\text{az}).

       For :attr:`.SamplingStrategy.SHELL_UNIFORM` the elevation distribution is
       chosen so the induced density on the sphere is area-uniform over the
       cap - see [Wolfram Mathworld :: Sphere point picking](https://mathworld.wolfram.com/SpherePointPicking.html).

    2. Transform the sampled offsets from the rig frame of ``last_pose`` into
       the VIO world frame using :class:`efm3d.aria.PoseTW`.

    3. Use :func:`oracle_rri.utils.frames.view_axes_from_points` to construct
       an RDF camera rotation for each sampled centre, with the camera
       looking back at ``last_pose``.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Populate ``ctx['poses']`` with shell-sampled candidate poses.

        Args:
            ctx: CandidateContext with at least ``last_pose`` and ``device``
                fields initialised.

        Returns:
            Updated context with:

            * ``poses``: :class:`efm3d.aria.PoseTW` of shape ``(N,)`` giving
              :math:`T_\\text{world,cam}` for each candidate.
            * ``mask``: all-True boolean mask of length ``N``; pruning rules
              will refine this further.
        """
        cfg = self.config
        n = ctx["poses"].shape[0]
        dev = ctx["device"]
        # 1) Sample radii r ~ U[min_radius, max_radius].
        r = torch.rand(n, device=dev) * (cfg.max_radius - cfg.min_radius) + cfg.min_radius
        # 2) Sample azimuth angles, either full circle or half-sphere.
        az = torch.rand(n, device=dev) * (2 * torch.pi if cfg.azimuth_full_circle else torch.pi)
        # 3) Convert (elev, az) to unit directions on the spherical cap.
        pos_local = self._sample_directions(n, az, dev)
        # 4) Scale by radius to obtain offsets in the last-pose rig frame.
        pos_local = pos_local * r.unsqueeze(1)

        pose_last: PoseTW = ctx["last_pose"]
        # 5) Map local offsets into the VIO world frame.
        pos_world = pose_last.transform(pos_local)

        # 6) Orient each candidate camera to look back at the last pose using
        #    the shared RDF / VIO-aligned helper.
        look_at = pose_last.t.expand_as(pos_world)
        r_cam = view_axes_from_points(cam_pos=pos_world, look_at=look_at)
        poses_tw = PoseTW.from_Rt(r_cam, pos_world)
        ctx["poses"] = poses_tw
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=dev)
        return ctx

    def _sample_directions(self, n: int, az: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample unit directions on a spherical cap.

        This helper samples elevation :math:`\\theta` and azimuth :math:`\\phi`
        according to :class:`.SamplingStrategy`:

        * ``SHELL_UNIFORM`` draws :math:`u \\sim \\mathcal{U}(\\sin\\theta_{\\min},
          \\sin\\theta_{\\max})` and sets :math:`\\theta = \\arcsin u`, which is
          equivalent to sampling uniformly over surface area on the cap.
        * ``FORWARD_GAUSSIAN`` draws :math:`\\theta` from a truncated normal
          within the elevation band, biasing views towards the forward axis.

        Args:
            n: Number of directions to sample.
            az: Tensor of shape ``(n,)`` with azimuth angles in radians.
            device: Torch device for the returned tensor.

        Returns:
            Tensor of shape ``(n, 3)`` with unit direction vectors in the
            last-pose rig frame.
        """
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
    """Reject candidates that are too close to the GT mesh.

    This rule queries the signed distance from each candidate camera centre
    to the mesh surface using
    [:class:`trimesh.proximity.ProximityQuery`](https://trimesh.org/trimesh.proximity.html#trimesh.proximity.ProximityQuery.signed_distance)
    and enforces a minimum clearance ``min_distance_to_mesh``.

    Note:
        According to the trimesh documentation, signed distances are
        positive inside or near the surface and negative outside the mesh.
        We simply require ``dist > min_distance_to_mesh`` to mark a
        candidate as clear; this behaviour may be tuned in the future if
        tighter control over near-surface points is needed.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply minimum-distance filtering to the current candidates."""
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None or self.config.min_distance_to_mesh <= 0:
            ctx["mask"] = mask
            return ctx

        # Evaluate signed distance at each candidate camera centre.
        positions = ctx["poses"].t  # PoseTW.t -> (N,3)
        query = trimesh.proximity.ProximityQuery(mesh)
        dist = query.signed_distance(positions.detach().cpu().numpy())
        # Keep only candidates with distance strictly larger than the configured
        # clearance threshold.
        clear = torch.from_numpy(dist).to(mask.device) > self.config.min_distance_to_mesh
        ctx["mask"] = mask & clear
        return ctx


class PathCollisionRule:
    """Reject candidates whose straight-line path intersects the mesh.

    For each candidate camera centre :math:`p_i` and the last pose origin :math:`o`, we construct a ray

    .. math::

        r_i(t) = o + t \\hat{d}_i, \\qquad
        \\hat{d}_i = \\frac{p_i - o}{\\lVert p_i - o \\rVert_2},

    and query for intersections up to ``max_distance = ||p_i - o||`` using
    either the generic [:mod:`trimesh.ray`](https://trimesh.org/ray.html) interface or
    [:class:`trimesh.ray.ray_pyembree.RayMeshIntersector`](https://trimesh.org/trimesh.ray.ray_pyembree.html#trimesh.ray.ray_pyembree.RayMeshIntersector).

    Candidates for which [:meth:`intersects_any`](https://trimesh.org/trimesh.ray.ray_triangle.html#trimesh.ray.ray_triangle.RayMeshIntersector.intersects_any) reports a hit are marked as colliding and removed from the mask.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply path-collision filtering between last pose and candidates."""
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None:
            ctx["mask"] = mask
            return ctx
        # Build a ray segment from the last pose origin to each candidate
        # camera centre.
        origin = ctx["last_pose"].t.view(1, 3)
        poses = ctx["poses"]
        targets = poses.t  # (N,3)
        dirs = targets - origin
        dists = torch.linalg.norm(dirs, dim=1).clamp(min=1e-6)
        dirs_norm = dirs / dists.unsqueeze(1)

        # Convert batched rays to numpy arrays for trimesh / PyEmbree.
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

        # Query whether each ray hits the mesh within the corresponding
        # max_distance; candidates with any hit are considered colliding.
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
    """Restrict candidates to a world-space axis-aligned bounding box.

    The rule assumes ``occupancy_extent`` encodes a 3D AABB in the VIO world
    frame as ``[xmin, xmax, ymin, ymax, zmin, zmax]`` and keeps only those
    candidate camera centres whose translations satisfy these bounds.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply AABB-based free-space filtering to current candidates."""
        mask = ctx["mask"]
        extent = ctx.get("occupancy_extent")
        if extent is None:
            ctx["mask"] = mask
            return ctx
        extent = extent.to(mask.device)
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        # Positions of candidate camera centres in world coordinates.
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


__all__ = [
    "Rule",
    "ShellSamplingRule",
    "MinDistanceToMeshRule",
    "PathCollisionRule",
    "FreeSpaceRule",
]

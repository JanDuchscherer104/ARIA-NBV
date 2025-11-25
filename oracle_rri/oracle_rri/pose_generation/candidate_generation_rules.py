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
``[0, 0, -g]`` and camera frames are LUF (X-left, Y-up, Z-forward).
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
    """Sample candidate camera poses on a spherical shell around the last pose.

    Conceptually, this rule proposes a cloud of viewpoints in three steps:

    1. Sample radii :math:`r \\sim \\mathcal{U}(r_{\\min}, r_{\\max})` and
       directions on a spherical *cap* defined by ``min_elev_deg``,
       ``max_elev_deg``, and ``azimuth_full_circle``. Directions are
       parameterised as

       .. math::

           x &= \\cos(\\text{elev}) \\cos(\\text{az}), \\\\
           y &= \\sin(\\text{elev}), \\\\
           z &= \\cos(\\text{elev}) \\sin(\\text{az}).

       For :attr:`.SamplingStrategy.SHELL_UNIFORM` the elevation distribution
       is chosen such that the induced density on the cap is approximately
       area-uniform (see `Sphere point picking`_).

    2. Transform the sampled offsets from the last-pose rig frame into the VIO
       world frame using :class:`efm3d.aria.PoseTW`.

    3. Use :func:`oracle_rri.utils.frames.view_axes_from_points` to orient
       each candidate so the LUF camera frame looks back at ``last_pose``.

    This yields candidate :math:`T_{\\text{world,cam}}` poses that are easy to
    constrain via later rules (collision, distance to mesh, free space).

    .. _Sphere point picking: https://mathworld.wolfram.com/SpherePointPicking.html
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Populate ``ctx['poses']`` with shell-sampled candidate poses.

        Args:
            ctx: Mutable :class:`.CandidateContext` with at least
                ``last_pose`` (``PoseTW``), ``device`` (``torch.device``) and
                a preallocated ``poses`` tensor-like container of length ``N``.

        Returns:
            CandidateContext: Updated context with:

            * ``poses``: :class:`efm3d.aria.PoseTW` of shape ``(N,)`` holding
              :math:`T_{\\text{world,cam}}` for each candidate.
            * ``mask``: Boolean tensor of shape ``(N,)`` initialised to all
              ``True``; follow-up rules refine this mask in-place.
        """
        cfg = self.config
        n = ctx["poses"].shape[0]
        dev = ctx["device"]
        # 1) Sample radii r ~ U[min_radius, max_radius].
        r = torch.rand(n, device=dev) * (cfg.max_radius - cfg.min_radius) + cfg.min_radius
        # 2) Sample azimuth angles, optionally clamped by camera FOV.
        az_half_deg = ctx.get("azimuth_half_range_deg")
        if az_half_deg is not None:
            half_rad = torch.deg2rad(torch.tensor(az_half_deg, device=dev))
            az_center = torch.pi / 2  # forward axis (z) in LUF frame
            az = az_center + (torch.rand(n, device=dev) - 0.5) * 2 * half_rad
        else:
            az = torch.rand(n, device=dev) * (2 * torch.pi if cfg.azimuth_full_circle else torch.pi)
        # 3) Convert (elev, az) to unit directions on the spherical cap using per-call overrides.
        min_elev_deg = ctx.get("min_elev_deg", cfg.min_elev_deg)
        max_elev_deg = ctx.get("max_elev_deg", cfg.max_elev_deg)
        pos_local = self._sample_directions(n, az, dev, min_elev_deg=min_elev_deg, max_elev_deg=max_elev_deg)
        # 4) Scale by radius to obtain offsets in the last-pose rig frame.
        pos_local = pos_local * r.unsqueeze(1)

        pose_last: PoseTW = ctx["last_pose"]
        # 5) Map local offsets into the VIO world frame.
        pos_world = pose_last.transform(pos_local)

        # 6) Orient each candidate camera to look back at the last pose using
        #    the shared LUF / VIO-aligned helper.
        look_at = pose_last.t.expand_as(pos_world)
        r_cam = view_axes_from_points(cam_pos=pos_world, look_at=look_at)
        poses_tw = PoseTW.from_Rt(r_cam, pos_world)
        ctx["poses"] = poses_tw
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=dev)
        return ctx

    def _sample_directions(
        self,
        n: int,
        az: torch.Tensor,
        device: torch.device,
        *,
        min_elev_deg: float | torch.Tensor | None = None,
        max_elev_deg: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample unit directions on a spherical cap in the rig frame.

        Elevation :math:`\\theta` and azimuth :math:`\\phi` are drawn according
        to :class:`.SamplingStrategy`:

        * ``SHELL_UNIFORM`` draws :math:`u \\sim \\mathcal{U}(\\sin\\theta_{\\min},
          \\sin\\theta_{\\max})` and sets :math:`\\theta = \\arcsin u`, which
          induces an approximately uniform density over surface area on the
          cap.
        * ``FORWARD_GAUSSIAN`` samples :math:`\\theta` from a truncated normal
          within the elevation band, biasing samples towards the band centre.

        Args:
            n: Number of directions to sample.
            az: Tensor of shape ``(n,)`` with azimuth angles in radians.
            device: Torch device on which to allocate the result.

        Returns:
            Tensor: Float tensor of shape ``(n, 3)`` with unit direction
            vectors in the last-pose rig frame (LUF convention).
        """
        min_elev = torch.deg2rad(
            torch.tensor(self.config.min_elev_deg if min_elev_deg is None else min_elev_deg, device=device)
        )
        max_elev = torch.deg2rad(
            torch.tensor(self.config.max_elev_deg if max_elev_deg is None else max_elev_deg, device=device)
        )

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
    """Reject candidates whose camera centres are too close to the GT mesh.

    For each candidate camera centre :math:`p_i`, this rule queries the signed
    distance to the mesh surface and enforces a minimum clearance
    ``min_distance_to_mesh`` in world coordinates.

    * CPU path: uses :class:`trimesh.proximity.ProximityQuery.signed_distance`
      on the full :class:`trimesh.Trimesh`.
    * GPU path: when ``collision_backend`` is ``P3D`` or ``EFM3D`` and a
      vertex/face representation is available in the context, distances are
      computed on-device via :func:`_point_mesh_distance_efm3d` to avoid
      host-device transfers.

    Distances are interpreted following the trimesh convention (positive near
    or inside the surface, negative outside). Candidates with
    ``dist > min_distance_to_mesh`` are marked as clear; others are masked out.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply minimum-distance filtering to the current candidates.

        Args:
            ctx: Mutable :class:`.CandidateContext` with fields:
                ``poses`` (``PoseTW`` of shape ``(N,)``), ``mask`` (``(N,)``),
                and either ``gt_mesh`` (``trimesh.Trimesh``) or mesh vertices
                and faces (``mesh_verts``, ``mesh_faces``) for the GPU path.

        Returns:
            CandidateContext: Same mapping with ``mask`` updated to keep only
            candidates whose distance to the mesh exceeds the configured
            clearance.
        """
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None or self.config.min_distance_to_mesh <= 0:
            ctx["mask"] = mask
            return ctx

        backend = ctx.get("collision_backend", self.config.collision_backend)
        positions = ctx["poses"].t  # (N,3)

        if backend in (CollisionBackend.P3D, CollisionBackend.EFM3D) and ctx.get("mesh_verts") is not None:
            verts = ctx["mesh_verts"]
            faces = ctx["mesh_faces"]
            dists = point_mesh_distance(
                positions,
                verts,
                faces,
            )
            ctx["mask"] = mask & (dists > self.config.min_distance_to_mesh)
            return ctx

        # Evaluate signed distance at each candidate camera centre.
        query = trimesh.proximity.ProximityQuery(mesh)
        dist = query.signed_distance(positions.detach().cpu().numpy())
        # Keep only candidates with distance strictly larger than the configured
        # clearance threshold.
        clear = torch.from_numpy(dist).to(mask.device) > self.config.min_distance_to_mesh
        ctx["mask"] = mask & clear
        return ctx


class PathCollisionRule:
    """Reject candidates whose straight-line path from the last pose hits the mesh.

    For each candidate centre :math:`p_i` and last-pose origin :math:`o`, we
    consider the ray segment

    .. math::

        r_i(t) = o + t \\hat{d}_i, \\qquad
        \\hat{d}_i = \\frac{p_i - o}{\\lVert p_i - o \\rVert_2},

    and test for intersections up to ``max_distance = ||p_i - o||``.

    * CPU path: uses the generic :mod:`trimesh.ray` interface or an optional
      :class:`trimesh.ray.ray_pyembree.RayMeshIntersector` backend and checks
      whether :meth:`intersects_any` reports a hit.
    * GPU path: for ``collision_backend`` in {``P3D``, ``EFM3D``} and when
      ``mesh_verts`` / ``mesh_faces`` are present, the ray segment is
      discretised into a small number of samples. Each sample is tested
      against the triangle mesh via :func:`_point_mesh_distance_efm3d`, and
      candidates with points closer than ``step_clearance`` are rejected.

    This rule ensures the straight-line motion from the current pose to a
    candidate pose does not pass through solid geometry.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply path-collision filtering between last pose and candidates.

        Args:
            ctx: Mutable :class:`.CandidateContext` with fields:
                ``last_pose`` (``PoseTW``), ``poses`` (``PoseTW`` of shape
                ``(N,)``), ``mask`` (``(N,)``), and either ``gt_mesh``
                (``trimesh.Trimesh``) or ``mesh_verts`` / ``mesh_faces`` for
                the GPU path.

        Returns:
            CandidateContext: Same mapping with ``mask`` updated to exclude
            candidates whose path from ``last_pose`` intersects the mesh.
        """
        mask = ctx["mask"]
        mesh: trimesh.Trimesh | None = ctx.get("gt_mesh")
        if mesh is None:
            ctx["mask"] = mask
            return ctx
        backend = ctx.get("collision_backend", self.config.collision_backend)
        # Build a ray segment from the last pose origin to each candidate
        # camera centre.
        origin = ctx["last_pose"].t.view(1, 3)
        poses = ctx["poses"]
        targets = poses.t  # (N,3)
        dirs = targets - origin
        dists = torch.linalg.norm(dirs, dim=1).clamp(min=1e-6)
        dirs_norm = dirs / dists.unsqueeze(1)

        if backend in (CollisionBackend.P3D, CollisionBackend.EFM3D) and ctx.get("mesh_verts") is not None:
            verts = ctx["mesh_verts"]
            faces = ctx["mesh_faces"]
            steps = max(2, int(self.config.ray_subsample))
            start = max(self.config.step_clearance / dists.max().item(), 1e-3)
            start = min(start, 1.0)
            t_vals = torch.linspace(start, 1.0, steps, device=targets.device)
            pts = origin.view(1, 1, 3) + dirs_norm.unsqueeze(1) * (t_vals.view(1, -1, 1) * dists.view(-1, 1, 1))
            pts_flat = pts.reshape(-1, 3)
            dists_pts = point_mesh_distance(
                pts_flat,
                verts,
                faces,
            ).view(targets.shape[0], steps)
            collide = (dists_pts < self.config.step_clearance).any(dim=1)
            ctx["mask"] = mask & (~collide)
            return ctx

        # Convert batched rays to numpy arrays for trimesh / PyEmbree (CPU fallback).
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
    """Restrict candidate camera centres to a world-space axis-aligned box.

    The rule assumes ``occupancy_extent`` encodes a 3D axis-aligned bounding
    box (AABB) in the VIO world frame as
    ``[xmin, xmax, ymin, ymax, zmin, zmax]`` and keeps only those candidate
    camera centres whose translations satisfy these bounds.

    In practice, these bounds are typically derived from semi-dense SLAM
    volume metadata or GT mesh bounds (see
    :func:`CandidateViewGenerator._occupancy_extent_from_sample` and the data
    pipeline docs in ``docs/contents/impl/data_pipeline_overview.qmd``).
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config

    def __call__(self, ctx: CandidateContext) -> CandidateContext:
        """Apply AABB-based free-space filtering to current candidates.

        Args:
            ctx: Mutable :class:`.CandidateContext` with fields:
                ``poses`` (``PoseTW`` of shape ``(N,)``), ``mask`` (``(N,)``),
                and ``occupancy_extent`` (tensor-like of shape ``(6,)`` with
                bounds ``[xmin, xmax, ymin, ymax, zmin, zmax]``) if available.

        Returns:
            CandidateContext: Same mapping with ``mask`` updated to only keep
            candidates whose world-space positions lie inside the AABB.
        """
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


def point_mesh_distance(
    points: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute point-to-mesh distances using PyTorch3D.

    Args:
        points: Tensor of shape ``(N, 3)`` with candidate query points in
            world coordinates.
        verts: Tensor of shape ``(V, 3)`` with mesh vertices.
        faces: Tensor of shape ``(F, 3)`` with long integer vertex indices.

    Returns:
        Tensor: Float tensor of shape ``(N,)`` with the minimal distance from
        each query point to the triangle mesh.
    """

    # Lazily import to avoid PyTorch3D dependency when the GPU path is unused.
    from pytorch3d.loss.point_mesh_distance import _DEFAULT_MIN_TRIANGLE_AREA, point_face_distance

    device = points.device
    points = points.to(device)
    verts = verts.to(device)
    faces = faces.to(device)

    # Build packed triangle tensor expected by PyTorch3D kernels.
    tris = verts[faces]  # (F, 3, 3)
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
    "ShellSamplingRule",
    "MinDistanceToMeshRule",
    "PathCollisionRule",
    "FreeSpaceRule",
]

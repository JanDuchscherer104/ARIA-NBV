"""Candidate view generation with composable pruning rules.

This module implements a *sampling-and-prune* pipeline for Next-Best-View
planning in ASE scenes:

1. Sample candidate camera centres on a spherical shell around the latest
   pose, using either area-uniform or forward-biased distributions in
   spherical coordinates.
2. Orient each candidate to look *away* from the most recent pose using
   :func:`oracle_rri.utils.frames.view_axes_from_points`, consistent with the
   EFM3D LUF camera convention (X-left, Y-up, Z-forward) and zero roll via
   ``world_up``.
3. Apply a sequence of rules (see :mod:`oracle_rri.pose_generation.candidate_generation_rules`)
   that remove candidates which violate geometric constraints (too close to
   the mesh, path collisions, outside occupancy bounds, ...).

Frames:
    * **World**: VIO-aligned world frame with gravity in ``[0, 0, -g]``
      (Z-up). All positions are expressed in this frame.
    * **Cameras**: LUF convention, X-left, Y-up, Z-forward (Project Aria).

NOTE: Earlier revisions treated cameras as RDF and silently mirrored poses.
We now document the correct LUF convention and rely on display-only transforms
to align visuals without changing stored poses.

Poses are :class:`efm3d.aria.PoseTW` with :math:`T_\\text{world,cam}` mapping
points from the camera frame into the VIO world frame.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

import torch
import trimesh
from efm3d.aria import PoseTW
from power_spherical import HypersphericalUniform
from pydantic import AliasChoices, Field, field_validator, model_validator
from torch.nn import functional as F  # noqa: N812

from ..configs.path_config import PathConfig
from ..data.efm_views import EfmSnippetView
from ..data.mesh_cache import MeshProcessSpec, ProcessedMesh, load_or_process_mesh
from ..utils import BaseConfig, Console, Verbosity, select_device
from ..utils.frames import view_axes_from_points, world_up_tensor
from .candidate_generation_rules import FreeSpaceRule, MinDistanceToMeshRule, PathCollisionRule, Rule
from .types import CandidateContext, CandidateSamplingResult, CollisionBackend, SamplingStrategy


class CandidateViewGeneratorConfig(BaseConfig["CandidateViewGenerator"]):
    """Config for candidate generation.

    Attributes:
        target: Factory target for the config. Runtime class instantiated by
            :meth:`BaseConfig.setup_target`.
        num_samples: Number of candidate poses to sample per call to
            :meth:`CandidateViewGenerator.generate`.
        max_resamples: Maximum number of re-sampling rounds used to fill
            the requested ``num_samples`` after applying pruning rules.
        min_radius: Inner radius (metres) of the spherical shell from which
            candidate camera centres are drawn (around ``last_pose``).
        max_radius: Outer radius (metres) of the spherical shell.
        min_elev_deg: Minimum elevation angle (degrees) for sampled view
            directions, measured in the last-pose rig frame.
        max_elev_deg: Maximum elevation angle (degrees) for sampled view
            directions.
        azimuth_full_circle: If ``True``, sample azimuth uniformly over
            :math:`[0, 2\\pi)`; otherwise only over a half-sphere about the
            forward axis.
        sampling_strategy: Distribution for yaw/pitch sampling, see
            :class:`oracle_rri.pose_generation.types.SamplingStrategy`.
        ensure_collision_free: Enable mesh-based straight-line path
            filtering using :class:`PathCollisionRule`.
        collision_backend: Ray backend, choosing between pure trimesh and
            :mod:`trimesh.ray.ray_pyembree`.
        min_distance_to_mesh: Clearance (metres) enforced between candidate
            camera centres and the GT mesh via
            :class:`MinDistanceToMeshRule`.
        ensure_free_space: Enable voxel-extent filtering via
            :class:`FreeSpaceRule`.
        ray_subsample: Number of ray samples / subdivisions when evaluating
            path collisions (currently unused but reserved for finer
            segment-based checks).
        step_clearance: Step size (metres) for distance checks along paths
            when needed.
        device: Torch device used for vectorised ops (e.g. ``"cuda"`` or
            ``"cpu"``); in debug mode this is forced to CPU.
    """

    target: type["CandidateViewGenerator"] = Field(default_factory=lambda: CandidateViewGenerator, exclude=True)

    """Factory target for the config. Runtime class instantiated by `setup_target()`.
    This is excluded from serialization."""

    camera_index: Literal["rgb", "slaml", "slamr"] = "rgb"
    """Camera index to use for candidate generation."""

    num_samples: int = 512
    """Number of candidate poses to return per `generate` call."""

    oversample_factor: float = 2.0
    """Oversampling multiplier before pruning (guards against rejections)."""

    max_resamples: int = 3
    """Deprecated: kept for config compatibility; oversampling is preferred."""
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

    delta_azimuth_deg: float = 360.0
    """Explicit azimuth span (degrees) centred on rig-forward.

    Example: 360 → full sphere, 180 → half-sphere, 90 → ±45° about forward.
    Takes precedence over ``azimuth_full_circle`` when < 360.
    """
    sampling_strategy: SamplingStrategy = SamplingStrategy.SHELL_UNIFORM
    """Distribution strategy for yaw/pitch sampling (area-uniform or forward-biased)."""

    kappa: float = 4.0
    """Concentration for PowerSpherical forward sampling (higher = tighter)."""

    ensure_collision_free: bool = True
    """Enable straight-line path collision filtering.

    When True, `_rule_path_collision` casts a ray from the last camera pose
    to each candidate viewpoint and rejects candidates whose path intersects
    the GT meshs.
    """

    collision_backend: CollisionBackend = CollisionBackend.P3D
    """Backend for ray/mesh intersection tests.

    `PYEMBREE` uses `RayMeshIntersector` (if installed) for fast queries;
    `TRIMESH` falls back to the pure-Python trimesh ray engine.
    `P3D` uses a CUDA path built on PyTorch3D (sampled surface + KNN).
    """

    min_distance_to_mesh: float = 0.0
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

    world_up: torch.Tensor = Field(default_factory=lambda: world_up_tensor())
    """World up direction (defaults to +Z in VIO coordinates)."""

    device: torch.device = Field(  # type: ignore[assignment]
        default_factory=lambda: select_device("auto", component="CandidateViewGenerator")
    )
    """Preferred torch device for vectorised operations (auto-resolves to CUDA when available)."""

    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging (0=quiet, 1=normal, 2=verbose).",
    )

    is_debug: bool = False
    """If True, enable debug logging and set device to 'cpu'."""

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, v: str | torch.device) -> torch.device:
        return select_device(v, component="CandidateViewGenerator")

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Any) -> Verbosity:
        return Verbosity.from_any(value)

    @model_validator(mode="after")
    def set_debug(self) -> Self:
        if self.is_debug:
            object.__setattr__(self, "device", torch.device("cpu"))
            object.__setattr__(self, "verbosity", Verbosity.VERBOSE)
            Console.with_prefix(self.__class__.__name__, "set_debug").set_verbosity(Verbosity.VERBOSE).log(
                "Debug mode enabled: forcing device to 'cpu' and verbose logging."
            )
        return self


class CandidateViewGenerator:
    """Generate candidate :class:`PoseTW` around the latest pose using rules.

    The generator implements a simple Monte Carlo NBV scheme:

    1. Draw ``num_samples`` candidate camera centres on a spherical shell
       around the latest pose using :class:`ShellSamplingRule`.
    2. Optionally cull candidates that start too close to the mesh
       (:class:`MinDistanceToMeshRule`).
    3. Optionally cull candidates whose straight-line path intersects the
       mesh (:class:`PathCollisionRule`).
    4. Optionally restrict candidates to a world-space AABB
       (:class:`FreeSpaceRule`).

    The result is a batch of candidate :math:`T_\\text{world,cam}` poses
    expressed in the VIO world frame, ready for oracle RRI evaluation or
    backbone feature extraction.

    Background:
        - NBV formulation and RRI scoring: ``docs/contents/theory/nbv_background.qmd``,
          ``docs/contents/impl/rri_computation.qmd``.
        - Pose math follows the Aria LUF convention as implemented in
          :mod:`efm3d.aria.pose`; rotations are assembled via
          :func:`oracle_rri.utils.frames.view_axes_from_points`.
        - Collision backends:
            * CPU: trimesh / pyembree rays.
            * GPU: PyTorch3D distance kernel (see
              [PyTorch3D :: point_mesh_face_distance](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance))
              or efm3d torch geometry utilities.
    """

    def __init__(self, config: CandidateViewGeneratorConfig):
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self.rules: list[Rule] = []
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """Construct the default rule sequence based on the config."""
        # Sampling happens centrally; rules only prune.
        if self.config.min_distance_to_mesh > 0:
            self.rules.append(MinDistanceToMeshRule(self.config))
        if self.config.ensure_collision_free:
            self.rules.append(PathCollisionRule(self.config))
        if self.config.ensure_free_space:
            self.rules.append(FreeSpaceRule(self.config))

    def generate_from_typed_sample(self, sample: EfmSnippetView) -> CandidateSamplingResult:
        """Generate candidate poses using data from an :class:`EfmSnippetView`."""
        device = torch.device(self.config.device)
        occ = sample.get_occupancy_extend()
        occupancy_extent = occ.to(device=device, dtype=torch.float32) if occ is not None else None
        gt_mesh = sample.mesh
        mesh_verts = sample.mesh_verts
        mesh_faces = sample.mesh_faces

        # Reuse processed mesh cache when possible.
        if gt_mesh is not None:
            bounds_min = (
                occupancy_extent[[0, 2, 4]] if occupancy_extent is not None else torch.as_tensor(gt_mesh.bounds[0])
            )
            bounds_max = (
                occupancy_extent[[1, 3, 5]] if occupancy_extent is not None else torch.as_tensor(gt_mesh.bounds[1])
            )
            spec = MeshProcessSpec(
                scene_id=sample.scene_id,
                snippet_id=sample.snippet_id,
                bounds_min=bounds_min.tolist(),
                bounds_max=bounds_max.tolist(),
                margin_m=0.2,
                simplify_ratio=None,
                max_faces=None,
                crop_min_keep_ratio=0.05,
            )
            proc: ProcessedMesh = load_or_process_mesh(
                gt_mesh,
                spec=spec,
                paths=PathConfig(),
                console=self.console,
            )
            gt_mesh = proc.mesh
            mesh_verts = proc.verts.to(device=device)
            mesh_faces = proc.faces.to(device=device)

        cam_view = sample.get_camera(self.config.camera_index)

        fov = cam_view.get_fov().to(device=device)  # F, 2
        self.console.dbg_summary("camera_fov", fov)
        self.console.plog({"fov": fov})

        last_pose = sample.trajectory.final_pose.to(device=device)
        self.console.log(
            f"generate_from_typed_sample scene={sample.scene_id} snippet={sample.snippet_id} mesh={gt_mesh is not None}"
        )
        self.console.log(
            "config: "
            f"sampling_strategy={self.config.sampling_strategy.name}, "
            f"collision_backend={self.config.collision_backend.name}, "
            f"device={self.config.device}"
        )
        if occupancy_extent is not None:
            self.console.log_summary("occupancy_extent", occupancy_extent)
        if gt_mesh is not None:
            self.console.log(f"mesh stats verts={gt_mesh.vertices.shape[0]:,} faces={gt_mesh.faces.shape[0]:,}")

        return self.generate(
            last_pose=last_pose,
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            occupancy_extent=occupancy_extent,
            camera_fov=fov,
        )

    def generate(
        self,
        last_pose: PoseTW,
        gt_mesh: trimesh.Trimesh | None = None,
        mesh_verts: torch.Tensor | None = None,
        mesh_faces: torch.Tensor | None = None,
        occupancy_extent: torch.Tensor | None = None,
        camera_fov: torch.Tensor | None = None,
    ) -> CandidateSamplingResult:
        """Sample candidate poses around ``last_pose`` and apply pruning rules.

        This method performs up to ``max_resamples`` rounds of sampling to
        accumulate ``num_samples`` valid poses:

        * In each round, a fresh batch of candidates is seeded and passed
          through the configured rule sequence.
        * Per-rule boolean masks are collected and combined into a single
          ``mask_valid`` requiring a candidate to satisfy *all* rules.
        * Valid candidates are concatenated until the desired batch size
          is reached or the resampling budget is exhausted.

        Args:
            last_pose: Current rig pose (PoseTW) in the VIO world frame;
                candidate positions are sampled around its translation.
            gt_mesh: Optional GT mesh for collision and distance checks.
            occupancy_extent: Optional explicit ``[6]`` bounds overriding
                :attr:`CandidateViewGeneratorConfig.occupancy_extent`.
            camera_fov: Optional ``Tensor[F,2]`` (fov_x, fov_y in degrees) for
                the active camera. When provided, it clamps elevation/azimuth
                sampling to the actual optics of the selected camera.

        Returns:
            CandidateSamplingResult dictionary with:

            * ``poses`` - PoseTW of valid candidates (possibly padded).
            * ``mask_valid`` - Boolean mask over ``poses`` indicating which
              entries passed all rules.
            * ``masks`` - Stacked per-rule masks aligned with ``shell_poses``.
            * ``rule_names`` - Names matching ``masks`` first dimension.
            * ``shell_poses`` - Raw shell-sampled poses before masking,
              useful for visualisation.
        """
        device = self.config.device
        effective_backend = self.config.collision_backend

        if gt_mesh is not None:
            if mesh_verts is None:
                mesh_verts = torch.as_tensor(gt_mesh.vertices, device=device, dtype=torch.float32)
            else:
                mesh_verts = mesh_verts.to(device=device, dtype=torch.float32)
            if mesh_faces is None:
                mesh_faces = torch.as_tensor(gt_mesh.faces, device=device, dtype=torch.int64)
            else:
                mesh_faces = mesh_faces.to(device=device, dtype=torch.int64)

            if device.type == "cpu" and self.config.collision_backend == CollisionBackend.P3D:
                self.console.warn("P3D collision backend requested on CPU; This is very slow!")

        # Derive per-call sampling limits from camera FOV if available to keep
        # sampling consistent with the selected camera optics.
        min_elev_deg = self.config.min_elev_deg
        max_elev_deg = self.config.max_elev_deg
        az_half_range_deg: float | None = None
        if camera_fov is not None and camera_fov.numel() > 0:
            fov_flat = camera_fov.reshape(-1, 2)
            valid = torch.isfinite(fov_flat).all(dim=1)
            if valid.any():
                fov_vals = fov_flat[valid][0]
                fov_x_deg = float(fov_vals[0].item())
                fov_y_deg = float(fov_vals[1].item())
                half_v = fov_y_deg / 2.0
                min_elev_deg = max(min_elev_deg, -half_v)
                max_elev_deg = min(max_elev_deg, half_v)
                az_half_range_deg = fov_x_deg / 2.0

        # Apply explicit azimuth delta if set (<360)
        if self.config.delta_azimuth_deg < 360:
            az_cfg_half = self.config.delta_azimuth_deg / 2.0
            az_half_range_deg = az_cfg_half if az_half_range_deg is None else min(az_half_range_deg, az_cfg_half)
        elif not self.config.azimuth_full_circle:
            az_half_range_deg = math.degrees(math.pi / 2) if az_half_range_deg is None else az_half_range_deg

        # ------------------------------------------------------------------ sampling (positions unbiased)
        n_seed = int(math.ceil(self.config.num_samples * self.config.oversample_factor))
        self.console.log(f"Sampling {n_seed} raw candidates (target {self.config.num_samples}) on {device}")
        poses_raw = self._sample_candidates(
            last_pose=last_pose.to(device),
            n=n_seed,
            min_elev_deg=min_elev_deg,
            max_elev_deg=max_elev_deg,
            azimuth_half_range_deg=az_half_range_deg,
        )

        # ------------------------------------------------------------------ filtering
        ctx: CandidateContext = {
            "last_pose": last_pose.to(device),
            "gt_mesh": gt_mesh,
            "mesh_verts": mesh_verts,
            "mesh_faces": mesh_faces,
            "occupancy_extent": occupancy_extent if occupancy_extent is not None else self.config.occupancy_extent,
            "device": device,
            "poses": poses_raw,
            "mask": torch.ones(len(poses_raw), dtype=torch.bool, device=device),
            "collision_backend": effective_backend,
            "min_elev_deg": min_elev_deg,
            "max_elev_deg": max_elev_deg,
            "azimuth_half_range_deg": az_half_range_deg,
        }

        masks: list[torch.Tensor] = []
        for rule in self.rules:
            ctx = rule(ctx)
            masks.append(ctx["mask"])

        masks_stacked = (
            torch.stack(masks, dim=0) if masks else torch.ones(1, len(poses_raw), device=device, dtype=torch.bool)
        )
        mask_valid = masks_stacked.all(dim=0)
        kept_idx = torch.where(mask_valid)[0][: self.config.num_samples]

        if kept_idx.numel() == 0:
            self.console.warn("All candidate poses were rejected; returning zero-padded PoseTW batch.")
            poses_valid = PoseTW(torch.zeros(self.config.num_samples, 12, device=device))
            mask_valid_out = torch.zeros(self.config.num_samples, dtype=torch.bool, device=device)
            masks_out = masks_stacked[:, :0]
        else:
            pose_tensors = poses_raw._data if isinstance(poses_raw, PoseTW) else poses_raw
            pose_sel = pose_tensors[kept_idx]
            poses_valid = PoseTW(pose_sel)
            n_valid = pose_sel.shape[0]
            self.console.dbg_summary("poses_valid", poses_valid)
            mask_valid_out = torch.ones(n_valid, dtype=torch.bool, device=device)
            masks_out = masks_stacked[:, kept_idx]
            self.console.log(f"Generated {n_valid} valid candidates.")

        shell_poses = poses_raw._data if isinstance(poses_raw, PoseTW) else poses_raw

        return {
            "poses": poses_valid,
            "mask_valid": mask_valid_out,
            "masks": masks_out,
            "rule_names": [r.__class__.__name__ for r in self.rules],
            "shell_poses": shell_poses,
        }

    # ------------------------------------------------------------------ sampling helpers

    def _sample_candidates(
        self,
        *,
        last_pose: PoseTW,
        n: int,
        min_elev_deg: float,
        max_elev_deg: float,
        azimuth_half_range_deg: float | None,
    ) -> PoseTW:
        """Sample positions + orientations around ``last_pose``.

        Directions are drawn with `HypersphericalUniform` (uniform) or
        `PowerSpherical` (forward-biased) and then rejected if they fall
        outside the configured elevation/azimuth band. Each candidate is
        oriented to look *away* from ``last_pose`` with roll fixed by
        ``world_up``.
        """

        device = last_pose.device
        dirs = self._sample_directions(
            n=n,
            device=device,
            min_elev_deg=min_elev_deg,
            max_elev_deg=max_elev_deg,
            azimuth_half_range_deg=azimuth_half_range_deg,
        )

        # Radii (positions unbiased on shell)
        r = (
            torch.rand(len(dirs), device=device) * (self.config.max_radius - self.config.min_radius)
            + self.config.min_radius
        )
        offsets_rig = dirs * r.unsqueeze(1)

        # Transform to world frame
        pos_world = last_pose.transform(offsets_rig)

        # Orient to look away from last pose while keeping roll zero w.r.t world_up.
        # Optionally apply a PowerSpherical bias to orientation (not positions).
        view_dir = pos_world - last_pose.t.expand_as(pos_world)
        view_dir = F.normalize(view_dir, dim=1)

        if self.config.sampling_strategy == SamplingStrategy.FORWARD_GAUSSIAN and self.config.kappa > 0:
            dist = HypersphericalUniform(dim=3, device=device) if self.config.kappa == 0 else None
            # Draw biased directions around the outward vector using PowerSpherical.
            from power_spherical import PowerSpherical  # local import to avoid unused when not used

            dist = PowerSpherical(
                loc=view_dir,
                scale=torch.tensor(self.config.kappa, device=device, dtype=pos_world.dtype),
            )
            view_dir = dist.rsample()  # (N,3)
            view_dir = F.normalize(view_dir, dim=1)

        look_at = pos_world + view_dir
        r_wc = view_axes_from_points(
            from_pose=pos_world,
            look_at=look_at,
            world_up=self.config.world_up.to(device=device, dtype=pos_world.dtype),
        )

        poses_tw = PoseTW.from_Rt(r_wc, pos_world)
        return poses_tw

    def _sample_directions(
        self,
        *,
        n: int,
        device: torch.device,
        min_elev_deg: float,
        max_elev_deg: float,
        azimuth_half_range_deg: float | None,
        max_trials: int = 6,
    ) -> torch.Tensor:
        """Sample unit directions in rig frame using power_spherical distributions."""

        collected: list[torch.Tensor] = []
        min_elev = torch.deg2rad(torch.tensor(min_elev_deg, device=device))
        max_elev = torch.deg2rad(torch.tensor(max_elev_deg, device=device))
        az_half = math.pi if azimuth_half_range_deg is None else math.radians(azimuth_half_range_deg)

        dist = HypersphericalUniform(dim=3, device=device)

        for _ in range(max_trials):
            batch = max(n - sum(t.shape[0] for t in collected), 0)
            if batch <= 0:
                break
            # oversample within trial to counter rejection
            batch = int(batch * 1.5) + 4
            samples = dist.rsample((batch,))

            elev = torch.asin(samples[:, 1])
            az = torch.atan2(samples[:, 0], samples[:, 2])
            mask = (elev >= min_elev) & (elev <= max_elev)
            if az_half < math.pi:
                mask &= az.abs() <= az_half

            accepted = samples[mask]
            if accepted.numel():
                collected.append(accepted)
            if sum(t.shape[0] for t in collected) >= n:
                break

        if not collected:
            raise RuntimeError("Directional sampling failed to produce any candidates; relax elevation/azimuth bounds.")

        dirs = torch.cat(collected, dim=0)[:n]
        return dirs


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

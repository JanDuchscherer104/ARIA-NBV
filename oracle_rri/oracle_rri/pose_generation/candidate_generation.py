"""Candidate view generation with composable pruning rules.

This module implements a *sampling-and-prune* pipeline for Next-Best-View
planning in ASE scenes:

1. Sample candidate camera centres on a spherical shell around the latest
   pose, using either area-uniform or forward-biased distributions in
   spherical coordinates.
2. Orient each candidate to look back at the most recent pose using
   :func:`oracle_rri.utils.frames.view_axes_from_points`, consistent with the
   EFM3D LUF camera convention (X-left, Y-up, Z-forward).
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

from typing import Any, Literal, Self

import torch
import trimesh
from efm3d.aria import PoseTW
from pydantic import AliasChoices, Field, field_validator, model_validator
from pytorch3d.structures import Meshes  # type: ignore[import-untyped]

from ..data.efm_views import EfmSnippetView
from ..utils import BaseConfig, Console, Verbosity, select_device
from .candidate_generation_rules import (
    FreeSpaceRule,
    MinDistanceToMeshRule,
    PathCollisionRule,
    Rule,
    ShellSamplingRule,
)
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
    the GT meshs.
    """

    collision_backend: CollisionBackend = CollisionBackend.P3D
    """Backend for ray/mesh intersection tests.

    `PYEMBREE` uses `RayMeshIntersector` (if installed) for fast queries;
    `TRIMESH` falls back to the pure-Python trimesh ray engine.
    `P3D` uses a CUDA path built on PyTorch3D (sampled surface + KNN).
    `EFM3D` uses torch-only geometry utilities from efm3d (CUDA-capable).
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

    cache_meshes: bool = True
    """If True, cache per-mesh PyTorch3D structures for reuse within a process."""
    device: torch.device = Field(  # type: ignore[assignment]
        default_factory=lambda: select_device("auto", component="CandidateViewGenerator")
    )
    """Preferred torch device for vectorised operations (auto-resolves to CUDA when available)."""

    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL,
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
        self.rules = [ShellSamplingRule(self.config)]
        if self.config.min_distance_to_mesh > 0:
            self.rules.append(MinDistanceToMeshRule(self.config))
        if self.config.ensure_collision_free:
            self.rules.append(PathCollisionRule(self.config))
        if self.config.ensure_free_space:
            self.rules.append(FreeSpaceRule(self.config))

    def generate_from_typed_sample(self, sample: EfmSnippetView) -> CandidateSamplingResult:
        """Generate candidate poses using data from an :class:`EfmSnippetView`."""
        device = torch.device(self.config.device)
        occupancy_extent = sample.get_occupancy_extend().to(device=device, dtype=torch.float32)
        gt_mesh = sample.mesh

        cam_view = sample.get_camera(self.config.camera_index)

        fov = cam_view.get_fov().to(device=device)  # F, 2

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
            occupancy_extent=occupancy_extent,
            camera_fov=fov,
        )

    def generate(
        self,
        last_pose: PoseTW,
        gt_mesh: trimesh.Trimesh | None = None,
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
        mesh_p3d = None

        mesh_verts = None
        mesh_faces = None
        effective_backend = self.config.collision_backend
        if gt_mesh is not None:
            mesh_verts = torch.as_tensor(gt_mesh.vertices, device=device, dtype=torch.float32)
            mesh_faces = torch.as_tensor(gt_mesh.faces, device=device, dtype=torch.int64)
            mesh_p3d = self._mesh_structures(mesh_verts, mesh_faces, cache_key=id(gt_mesh))

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

        ctx: CandidateContext = {
            "last_pose": last_pose.to(device),
            "gt_mesh": gt_mesh,
            "mesh_p3d": mesh_p3d,
            "mesh_verts": mesh_verts,
            "mesh_faces": mesh_faces,
            "occupancy_extent": occupancy_extent if occupancy_extent is not None else self.config.occupancy_extent,
            "device": device,
            "poses": torch.empty(0, 12, device=device),
            "mask": torch.empty(0, dtype=torch.bool, device=device),
            "min_elev_deg": min_elev_deg,
            "max_elev_deg": max_elev_deg,
            "azimuth_half_range_deg": az_half_range_deg,
            "collision_backend": effective_backend,
        }

        poses_accum: list[torch.Tensor] = []
        masks_accum: list[torch.Tensor] = []  # NOTE: store per-round stacked masks for debugging
        shell_accum: list[PoseTW] = []
        remaining = self.config.num_samples
        attempts = 0
        self.console.log(
            f"Sampling {self.config.num_samples} candidates (max_resamples={self.config.max_resamples}) on {device}"
        )

        # Iteratively sample until we either collect ``num_samples`` valid
        # candidates or exhaust the resampling budget.
        while remaining > 0 and attempts < self.config.max_resamples:
            # Seed a fresh batch of candidate pose slots and full mask.
            ctx_batch = self._seed(ctx, remaining)
            masks: list[torch.Tensor] = []
            # Apply each rule in sequence; rules update ``poses`` and
            # refine ``ctx['mask']`` in-place.
            for rule in self.rules:
                ctx_batch = rule(ctx_batch)
                survivors = int(ctx_batch["mask"].sum().item())
                self.console.dbg(
                    f"Rule {rule.__class__.__name__}: {survivors}/{ctx_batch['mask'].numel()} candidates remain"
                )
                if survivors == 0:
                    self.console.warn(
                        f"Rule {rule.__class__.__name__} rejected all candidates in this round "
                        f"(attempt {attempts + 1}/{self.config.max_resamples})."
                    )
                masks.append(ctx_batch["mask"])
            # Preserve the raw sampled poses (before masking) for
            # visualisation of the full candidate space.
            shell_accum.append(ctx_batch["poses"].clone())
            mask_valid = (
                torch.stack(masks, dim=0).all(dim=0)
                if masks
                else torch.ones(ctx_batch["poses"].shape[0], dtype=torch.bool, device=device)
            )
            if mask_valid.any():
                poses_accum.append(ctx_batch["poses"][mask_valid])
            masks_accum.append(torch.stack(masks, dim=0))  # shape (num_rules, batch)
            # Update the remaining quota and round counter.
            remaining = self.config.num_samples - sum(p.shape[0] for p in poses_accum)
            attempts += 1
            self.console.dbg(f"Sampling attempt {attempts} complete; remaining quota {remaining}")

        if poses_accum:
            # `poses_accum` may contain `PoseTW` TensorWrappers; extract raw tensors
            # before concatenation to avoid passing TensorWrapper objects into
            # `PoseTW` again.
            pose_tensors = [
                p._data if isinstance(p, PoseTW) else p  # noqa: SLF001 - accessing protected attr for unwrap
                for p in poses_accum
            ]
            poses_cat = torch.cat(pose_tensors, dim=0)[: self.config.num_samples]
            mask_valid_out = torch.ones(poses_cat.shape[0], dtype=torch.bool, device=device)
        else:
            poses_cat = torch.zeros(self.config.num_samples, 12, device=device)
            mask_valid_out = torch.zeros(self.config.num_samples, dtype=torch.bool, device=device)
        poses_valid = PoseTW(poses_cat if isinstance(poses_cat, torch.Tensor) else poses_cat._data)  # type: ignore[arg-type]  # noqa: SLF001
        shell_poses = (
            torch.cat([p._data for p in shell_accum], dim=0) if shell_accum else torch.zeros(0, 12, device=device)
        )
        # NOTE: make per-rule masks align with shell_poses for debugging which rule killed which sample.
        masks_stacked = torch.cat(masks_accum, dim=1) if masks_accum else torch.zeros(len(self.rules), 0, device=device)
        survivor_count = int(mask_valid_out.sum().item())
        if survivor_count == 0:
            self.console.warn("All candidate poses were rejected; returning zero-padded PoseTW batch.")
        else:
            self.console.log(f"Generated {survivor_count} valid candidates.")
            self.console.dbg_summary("valid_pose_tensor", poses_cat[:survivor_count])
        return {
            "poses": poses_valid,
            "mask_valid": mask_valid_out,
            "masks": masks_stacked,
            "rule_names": [r.__class__.__name__ for r in self.rules],
            "shell_poses": shell_poses,
        }

    # ------------------------------------------------------------------ rules

    def _mesh_structures(
        self, mesh_verts: torch.Tensor, mesh_faces: torch.Tensor, cache_key: int
    ) -> tuple[Meshes | None, torch.Tensor | None]:
        """Return (Meshes, sampled_points) for PyTorch3D backend with caching."""
        # TODO: should be handeled by sample so that we don't have to convert it multiple times
        if self.config.collision_backend != CollisionBackend.P3D:
            return None, None

        if self.config.cache_meshes:
            cache = getattr(self, "_mesh_cache", None)
            if cache is None:
                cache = {}
                self._mesh_cache = cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        mesh_p3d = Meshes(verts=[mesh_verts], faces=[mesh_faces])

        if self.config.cache_meshes:
            self._mesh_cache[cache_key] = mesh_p3d

        return mesh_p3d

    def _seed(self, ctx: CandidateContext, n: int) -> CandidateContext:
        """Initialise pose tensor placeholders for a fresh sampling round.

        This resets ``ctx['poses']`` to a zero tensor of shape ``(n, 12)``,
        representing flattened :math:`[R \\mid t]` pose blocks, and sets
        ``ctx['mask']`` to an all-True mask. Subsequent rules will fill
        the pose tensor and progressively refine the mask.
        """
        ctx["poses"] = torch.zeros(n, 12, device=ctx["device"])
        ctx["mask"] = torch.ones(n, dtype=torch.bool, device=ctx["device"])
        return ctx


__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "SamplingStrategy",
    "CollisionBackend",
]

"""Mixed finite-candidate view generation with provenance."""

from __future__ import annotations

from typing import Any

import torch
import trimesh  # type: ignore[import-untyped]
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator, model_validator

from ..data_handling import EfmSnippetView
from ..utils import BaseConfig
from .candidate_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from .types import (
    CandidateGenerationRuntimeContext,
    CandidateSamplingResult,
    SamplingStrategy,
    ViewDirectionMode,
)

_STRATEGY_IDS = {
    ViewDirectionMode.FORWARD_RIG: 0,
    ViewDirectionMode.RADIAL_AWAY: 1,
    ViewDirectionMode.RADIAL_TOWARDS: 2,
    ViewDirectionMode.TARGET_POINT: 3,
}


def candidate_strategy_id(strategy: ViewDirectionMode | str) -> int:
    """Return the stable integer provenance id for a view-direction family."""

    return _STRATEGY_IDS[ViewDirectionMode(strategy)]


class CandidateMixtureComponentConfig(BaseConfig):
    """One fixed-count candidate-family component inside a mixture sampler."""

    name: str
    """Human-readable component name retained as row provenance."""

    count: int
    """Number of full-shell candidates sampled by this component."""

    strategy: ViewDirectionMode
    """View-direction family used as the stable candidate-strategy provenance."""

    sampling_strategy: SamplingStrategy | None = None
    """Optional positional sampling override."""

    view_sampling_strategy: SamplingStrategy | None = None
    """Optional view-direction sampling override."""

    min_radius: float | None = None
    max_radius: float | None = None
    min_elev_deg: float | None = None
    max_elev_deg: float | None = None
    delta_azimuth_deg: float | None = None
    kappa: float | None = None
    view_kappa: float | None = None
    view_max_angle_deg: float | None = None
    view_max_azimuth_deg: float | None = None
    view_max_elevation_deg: float | None = None
    view_roll_jitter_deg: float | None = None

    @field_validator("count")
    @classmethod
    def _positive_count(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("Candidate mixture component count must be >= 1.")
        return int(value)


class CandidateMixtureViewGeneratorConfig(BaseConfig):
    """Config-as-factory for fixed-count mixed candidate tables."""

    @property
    def target(self) -> type["CandidateMixtureViewGenerator"]:
        return CandidateMixtureViewGenerator

    base: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    """Base generator settings shared by all mixture components."""

    components: list[CandidateMixtureComponentConfig] = Field(
        default_factory=lambda: [
            CandidateMixtureComponentConfig(
                name="target_point",
                count=24,
                strategy=ViewDirectionMode.TARGET_POINT,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
            ),
            CandidateMixtureComponentConfig(
                name="radial_towards",
                count=12,
                strategy=ViewDirectionMode.RADIAL_TOWARDS,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
            ),
            CandidateMixtureComponentConfig(
                name="radial_away",
                count=12,
                strategy=ViewDirectionMode.RADIAL_AWAY,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
            ),
            CandidateMixtureComponentConfig(
                name="forward_rig",
                count=12,
                strategy=ViewDirectionMode.FORWARD_RIG,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
            ),
        ]
    )
    """Ordered mixture components. Full-shell row order follows this list."""

    @model_validator(mode="after")
    def _nonempty_components(self) -> "CandidateMixtureViewGeneratorConfig":
        if not self.components:
            raise ValueError("Candidate mixture requires at least one component.")
        return self

    @property
    def total_count(self) -> int:
        """Total full-shell candidate budget across mixture components."""

        return sum(component.count for component in self.components)

    @property
    def device(self) -> torch.device:
        """Resolved base generator device."""

        return self.base.device

    @property
    def camera_label(self) -> str:
        """Base camera label used for typed-snippet generation."""

        return self.base.camera_label


class CandidateMixtureViewGenerator:
    """Generate a fixed-size candidate table from multiple sampling families."""

    def __init__(self, config: CandidateMixtureViewGeneratorConfig) -> None:
        self.config = config

    def generate_from_typed_sample(
        self,
        sample: EfmSnippetView,
        frame_index: int | None = None,
        runtime_context: CandidateGenerationRuntimeContext | None = None,
    ) -> CandidateSamplingResult:
        """Generate mixed candidates from an EFM snippet."""

        device = torch.device(self.config.base.device)
        occ = sample.get_occupancy_extend()
        occupancy_extent = occ.to(device=device, dtype=torch.float32)
        gt_mesh = sample.mesh
        mesh_verts = sample.mesh_verts
        mesh_faces = sample.mesh_faces
        if mesh_verts is None or mesh_faces is None:
            raise ValueError("Candidate mixture generation requires sample.mesh_verts and sample.mesh_faces.")

        cam_view = sample.get_camera(self.config.base.camera_label)
        if frame_index is None:
            frame_index = self.config.base.reference_frame_index
        if frame_index is None:
            reference_pose = sample.trajectory.final_pose.to(device=device)
        else:
            _cam_idx, traj_idx = cam_view.nearest_traj_indices(
                sample.trajectory.time_ns,
                [frame_index],
                default_last=True,
            )
            reference_pose = (
                sample.trajectory.final_pose.to(device=device)
                if traj_idx.numel() == 0
                else sample.trajectory.t_world_rig[traj_idx].to(device=device)
            )

        return self.generate(
            reference_pose=reference_pose,
            gt_mesh=gt_mesh,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            camera_calib_template=cam_view.calib,
            occupancy_extent=occupancy_extent,
            runtime_context=runtime_context,
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
        runtime_context: CandidateGenerationRuntimeContext | None = None,
    ) -> CandidateSamplingResult:
        """Generate one concatenated full-shell candidate table."""

        component_results: list[CandidateSamplingResult] = []
        component_names: list[str] = []
        for component_index, component in enumerate(self.config.components):
            component_cfg = self._component_config(component, component_index, runtime_context=runtime_context)
            result = CandidateViewGenerator(component_cfg).generate(
                reference_pose=reference_pose,
                gt_mesh=gt_mesh,
                mesh_verts=mesh_verts,
                mesh_faces=mesh_faces,
                camera_calib_template=camera_calib_template,
                occupancy_extent=occupancy_extent,
            )
            shell_count = int(result.mask_valid.reshape(-1).shape[0])
            device = result.mask_valid.device
            result.strategy_id = torch.full(
                (shell_count,),
                candidate_strategy_id(component.strategy),
                dtype=torch.int64,
                device=device,
            )
            result.mixture_id = torch.full((shell_count,), component_index, dtype=torch.int64, device=device)
            result.sampler_probability = torch.full(
                (shell_count,),
                1.0 / float(self.config.total_count),
                dtype=torch.float32,
                device=device,
            )
            result.component_name = tuple(component.name for _ in range(shell_count))
            component_results.append(result)
            component_names.extend([component.name] * shell_count)

        return _concat_results(component_results, component_name=tuple(component_names))

    def _component_config(
        self,
        component: CandidateMixtureComponentConfig,
        component_index: int,
        *,
        runtime_context: CandidateGenerationRuntimeContext | None,
    ) -> CandidateViewGeneratorConfig:
        target_point = self.config.base.view_target_point_world
        if component.strategy == ViewDirectionMode.TARGET_POINT:
            if runtime_context is None or runtime_context.target_center_world is None:
                raise ValueError("TARGET_POINT candidate components require runtime_context.target_center_world.")
            target_point = torch.as_tensor(runtime_context.target_center_world, dtype=torch.float32).reshape(3)

        updates: dict[str, Any] = {
            "num_samples": component.count,
            "oversample_factor": 1.0,
            "view_direction_mode": component.strategy,
            "view_target_point_world": target_point,
        }
        if self.config.base.seed is not None:
            updates["seed"] = int(self.config.base.seed) + component_index

        for field_name in (
            "sampling_strategy",
            "view_sampling_strategy",
            "min_radius",
            "max_radius",
            "min_elev_deg",
            "max_elev_deg",
            "delta_azimuth_deg",
            "kappa",
            "view_kappa",
            "view_max_angle_deg",
            "view_max_azimuth_deg",
            "view_max_elevation_deg",
            "view_roll_jitter_deg",
        ):
            value = getattr(component, field_name)
            if value is not None:
                updates[field_name] = value
        return self.config.base.model_copy(update=updates)


def _concat_results(
    results: list[CandidateSamplingResult],
    *,
    component_name: tuple[str, ...],
) -> CandidateSamplingResult:
    if not results:
        raise ValueError("Cannot concatenate an empty candidate-mixture result.")

    first = results[0]
    views = CameraTW(torch.cat([result.views.tensor() for result in results], dim=0))
    shell_poses = PoseTW(torch.cat([result.shell_poses.tensor() for result in results], dim=0))
    mask_valid = torch.cat([result.mask_valid.reshape(-1) for result in results], dim=0)
    shell_offsets = _cat_optional([result.shell_offsets_ref for result in results])
    strategy_id = _cat_required([result.strategy_id for result in results], "strategy_id")
    mixture_id = _cat_required([result.mixture_id for result in results], "mixture_id")
    sampler_probability = _cat_required([result.sampler_probability for result in results], "sampler_probability")
    masks = _concat_masks(results, mask_valid)
    extras = _concat_extras(results)

    return CandidateSamplingResult(
        views=views,
        reference_pose=first.reference_pose,
        mask_valid=mask_valid,
        masks=masks,
        shell_poses=shell_poses,
        shell_offsets_ref=shell_offsets,
        sampling_pose=first.sampling_pose,
        strategy_id=strategy_id,
        mixture_id=mixture_id,
        sampler_probability=sampler_probability,
        component_name=component_name,
        extras=extras,
    )


def _cat_optional(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    if any(value is None for value in values):
        return None
    return torch.cat([value for value in values if value is not None], dim=0)


def _cat_required(values: list[torch.Tensor | None], name: str) -> torch.Tensor:
    if any(value is None for value in values):
        raise ValueError(f"Candidate mixture component did not provide {name}.")
    return torch.cat([value for value in values if value is not None], dim=0)


def _concat_masks(results: list[CandidateSamplingResult], mask_valid: torch.Tensor) -> dict[str, torch.Tensor]:
    names = sorted({name for result in results for name in result.masks})
    output: dict[str, torch.Tensor] = {}
    for name in names:
        chunks = []
        for result in results:
            chunks.append(result.masks.get(name, result.mask_valid).reshape(-1))
        output[name] = torch.cat(chunks, dim=0).to(device=mask_valid.device)
    return output


def _concat_extras(results: list[CandidateSamplingResult]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for result in results:
        for name, value in result.extras.items():
            if torch.is_tensor(value):
                extras.setdefault(name, []).append(value)
    return {name: torch.cat(values, dim=0) for name, values in extras.items() if len(values) == len(results)}


__all__ = [
    "CandidateMixtureComponentConfig",
    "CandidateMixtureViewGenerator",
    "CandidateMixtureViewGeneratorConfig",
    "candidate_strategy_id",
]

"""Configuration models for the offline Rerun inspector.

The inspector follows the project config-as-factory pattern: the top-level
``RerunOfflineInspectorConfig`` owns the nested dataset, selection, output,
geometry, performance, and primitive toggles used by the CLI runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from aria_nbv.data_handling import VinOfflineDatasetConfig
from aria_nbv.utils import BaseConfig, Verbosity


class RerunInspectorDatasetConfig(BaseConfig):
    """Dataset reader configuration for offline Rerun inspection."""

    offline: VinOfflineDatasetConfig = Field(
        default_factory=lambda: VinOfflineDatasetConfig(
            return_format="sample",
            map_location="cpu",
            load_candidates=True,
            load_candidate_pcs=False,
            load_depths=False,
            load_gt_obbs=True,
            load_detected_obbs=True,
            load_trajectory_metadata=True,
        ),
    )
    """Immutable VIN offline dataset reader used by the inspector."""

    @model_validator(mode="after")
    def _force_read_only_sample_reader(self) -> "RerunInspectorDatasetConfig":
        """Keep the inspector on the read-only sample-returning CPU path."""

        self.offline.return_format = "sample"
        self.offline.map_location = BaseConfig._resolve_device("cpu")
        return self


class RerunInspectorSelectionConfig(BaseConfig):
    """Sample selection knobs for ``nbv-rerun-inspect``."""

    sample_key: str | None = None
    """Stable offline sample key. Highest precedence when provided."""

    scene_id: str | None = None
    """ASE scene identifier used with ``snippet_id`` when no sample key is set."""

    snippet_id: str | None = None
    """ASE snippet identifier used with ``scene_id`` when no sample key is set."""

    split: Literal["all", "train", "val"] = "val"
    """Split used for index-based selection."""

    index: int = 0
    """Zero-based index inside ``split`` when no higher-precedence selector is set."""

    rollout_context_mode: Literal["auto", "required", "off"] = "auto"
    """VIN context policy for rollout-Zarr inspection."""

    @field_validator("index")
    @classmethod
    def _validate_index(cls, value: int) -> int:
        """Reject negative split indices."""

        if int(value) < 0:
            raise ValueError("selection.index must be >= 0.")
        return int(value)

    @model_validator(mode="after")
    def _validate_scene_snippet_pair(self) -> "RerunInspectorSelectionConfig":
        """Require scene/snippet selectors to be supplied together."""

        if (self.scene_id is None) ^ (self.snippet_id is None):
            raise ValueError("selection.scene_id and selection.snippet_id must be provided together.")
        return self


class RerunInspectorOutputConfig(BaseConfig):
    """Rerun recording output configuration."""

    mode: Literal["save", "spawn", "connect"] = "save"
    """Output sink used before logging starts."""

    application_id: str = "aria-nbv-rerun-inspector"
    """Rerun application identifier."""

    recording_id: str | None = None
    """Optional stable Rerun recording id."""

    save_path: Path = Path(".logs") / "rerun" / "offline_inspector.rrd"
    """Destination recording path when ``mode='save'``."""

    connect_addr: str | None = None
    """Rerun gRPC endpoint when ``mode='connect'``."""

    spawn_port: int = 9876
    """Viewer port used when ``mode='spawn'``."""

    spawn_memory_limit: str = "75%"
    """Rerun viewer memory limit used for spawned viewers."""

    hide_welcome_screen: bool = True
    """Whether spawned viewers should hide the Rerun welcome screen."""

    @field_validator("spawn_port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        """Validate the configured viewer port."""

        port = int(value)
        if port <= 0 or port > 65535:
            raise ValueError("output.spawn_port must be in [1, 65535].")
        return port


class RerunInspectorGeometryConfig(BaseConfig):
    """Geometry rendering parameters for Rerun primitives."""

    frustum_scale: float = 0.35
    """Candidate frustum size in world units."""

    reference_axis_length: float = 0.45
    """Axis length used for the reference-pose transform."""

    semidense_radius: float = 0.015
    """Rerun point radius for semidense world points."""

    candidate_center_radius: float = 0.035
    """Rerun point radius for candidate centers."""

    candidate_point_radius: float = 0.01
    """Rerun point radius for optional candidate point clouds."""

    trajectory_radius: float = 0.02
    """Line radius for trajectory paths."""

    mesh_alpha: int = 51
    """Alpha channel for the GT mesh albedo factor in ``[0, 255]``."""

    @field_validator(
        "frustum_scale",
        "reference_axis_length",
        "semidense_radius",
        "candidate_center_radius",
        "candidate_point_radius",
        "trajectory_radius",
    )
    @classmethod
    def _validate_positive(cls, value: float) -> float:
        """Validate positive geometry scales."""

        scalar = float(value)
        if scalar <= 0:
            raise ValueError("geometry scales and radii must be > 0.")
        return scalar

    @field_validator("mesh_alpha")
    @classmethod
    def _validate_mesh_alpha(cls, value: int) -> int:
        """Validate the GT mesh alpha channel."""

        alpha = int(value)
        if alpha < 0 or alpha > 255:
            raise ValueError("geometry.mesh_alpha must be in [0, 255].")
        return alpha


class RerunInspectorPerformanceConfig(BaseConfig):
    """Performance and deterministic sampling knobs."""

    max_semidense_points: int = 50_000
    """Maximum semidense points to log after deterministic downsampling."""

    max_candidate_points: int = 20_000
    """Maximum optional candidate point-cloud points to log."""

    seed: int | None = 0
    """Seed used for deterministic downsampling."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Console verbosity for the inspector."""

    _coerce_verbosity = field_validator("verbosity", mode="before")(BaseConfig._coerce_verbosity)
    _non_negative_seed = field_validator("seed")(BaseConfig._validate_non_negative_seed)

    @field_validator("max_semidense_points", "max_candidate_points")
    @classmethod
    def _validate_non_negative_limit(cls, value: int) -> int:
        """Validate point-count limits."""

        limit = int(value)
        if limit < 0:
            raise ValueError("point-count limits must be >= 0.")
        return limit


class RerunInspectorCandidateConfig(BaseConfig):
    """Candidate camera and selected-detail logging policy."""

    subset_mode: Literal["all", "valid_only", "invalid_only", "top_k_oracle", "indices"] = "all"
    """Candidate subset to log as native Rerun camera entities."""

    subset_top_k: int = 5
    """Number of candidates used when ``subset_mode='top_k_oracle'``."""

    subset_indices: list[int] = Field(default_factory=list)
    """Explicit candidate indices used when ``subset_mode='indices'``."""

    selected_strategy: Literal["top_valid_oracle", "first_valid", "explicit_index"] = "top_valid_oracle"
    """Strategy used for the single candidate that receives depth/point details."""

    selected_index: int | None = None
    """Explicit selected candidate index, overriding ``selected_strategy`` when set."""

    @field_validator("subset_top_k")
    @classmethod
    def _validate_positive_top_k(cls, value: int) -> int:
        """Validate positive candidate top-k counts."""

        count = int(value)
        if count <= 0:
            raise ValueError("candidate.subset_top_k must be > 0.")
        return count

    @field_validator("subset_indices")
    @classmethod
    def _validate_subset_indices(cls, value: list[int]) -> list[int]:
        """Reject negative candidate subset indices."""

        indices = [int(item) for item in value]
        if any(item < 0 for item in indices):
            raise ValueError("candidate.subset_indices must be non-negative.")
        return indices

    @field_validator("selected_index")
    @classmethod
    def _validate_selected_index(cls, value: int | None) -> int | None:
        """Reject negative selected candidate indices."""

        if value is None:
            return None
        index = int(value)
        if index < 0:
            raise ValueError("candidate.selected_index must be >= 0.")
        return index


class RerunInspectorEfmVoxelConfig(BaseConfig):
    """EFM voxel-field visualization policy."""

    enabled: bool = True
    """Whether to log curated EFM voxel fields when a backbone output is loaded."""

    log_occ_pr: bool = True
    """Log occupancy probabilities as thresholded voxel-center points."""

    log_cent_pr: bool = True
    """Log centerness probabilities as thresholded voxel-center points."""

    log_cent_pr_nms: bool = True
    """Log NMS-filtered centerness probabilities as thresholded voxel-center points."""

    occ_threshold: float = 0.95
    """Minimum ``occ_pr`` value to log."""

    cent_threshold: float = 0.03
    """Minimum ``cent_pr`` value to log."""

    cent_nms_threshold: float = 0.01
    """Minimum ``cent_pr_nms`` value to log."""

    max_points_per_field: int = 10_000
    """Maximum logged voxel centers per EFM field after thresholding."""

    point_radius: float = 0.025
    """Rerun point radius for logged voxel centers."""

    @field_validator("occ_threshold", "cent_threshold", "cent_nms_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        """Validate probability thresholds."""

        threshold = float(value)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("EFM voxel thresholds must be in [0, 1].")
        return threshold

    @field_validator("max_points_per_field")
    @classmethod
    def _validate_non_negative_points(cls, value: int) -> int:
        """Validate non-negative voxel point limits."""

        limit = int(value)
        if limit < 0:
            raise ValueError("efm_voxels.max_points_per_field must be >= 0.")
        return limit

    @field_validator("point_radius")
    @classmethod
    def _validate_positive_radius(cls, value: float) -> float:
        """Validate positive voxel point radius."""

        radius = float(value)
        if radius <= 0.0:
            raise ValueError("efm_voxels.point_radius must be > 0.")
        return radius


class RerunInspectorAseKeyframeConfig(BaseConfig):
    """ASE camera keyframe visualization policy."""

    frame_policy: Literal["first_last"] = "first_last"
    """Frame policy for the curated ASE camera stream view."""


class RerunInspectorPrimitivesConfig(BaseConfig):
    """Primitive toggles for the Rerun recording."""

    log_semidense: bool = True
    """Log VIN semidense world points."""

    log_reference_pose: bool = True
    """Log the oracle reference pose."""

    log_candidate_frusta: bool = True
    """Log all candidate frusta."""

    log_top_oracle_frustum: bool = True
    """Log the candidate frustum with highest oracle RRI."""

    log_invalid_frusta: bool = True
    """Log invalid candidate frusta when validity masks are available."""

    log_candidate_centers: bool = True
    """Log candidate camera centers."""

    log_metadata: bool = True
    """Log sample metadata as a Rerun text document."""

    log_mesh: bool = True
    """Log a GT mesh only when the inventory and sample expose one."""

    log_candidate_points: bool = False
    """Log candidate point clouds only when the inventory and sample expose them."""

    log_gt_mesh: bool = True
    """Log compact or live-attached GT mesh when available."""

    log_gt_obbs: bool = True
    """Log compact or live-attached GT OBBs when available."""

    log_detected_obbs: bool = True
    """Log compact detected OBBs when available."""

    show_gt_obb_labels: bool = True
    """Show GT OBB labels directly in 3D; labels are always logged as metadata."""

    show_detected_obb_labels: bool = False
    """Show detected EFM OBB labels directly in 3D; labels are always logged as metadata."""

    log_gt_trajectory: bool = True
    """Log the snippet rig trajectory when available."""

    log_candidate_depths: bool = False
    """Log candidate depth diagnostics when available."""

    log_rgb_keyframes: bool = False
    """Log live-attached RGB keyframes when a raw EFM snippet is attached."""

    log_depth_keyframes: bool = False
    """Log live-attached depth keyframes when a raw EFM snippet is attached."""

    log_efm_voxels: bool = True
    """Log curated EFM voxel evidence when available."""


class RerunOfflineInspectorConfig(BaseConfig):
    """Top-level config-as-factory model for the offline Rerun inspector."""

    @property
    def target(self) -> type[Any]:
        """Return the inspector runtime factory target."""

        from ._cli import RerunOfflineInspector

        return RerunOfflineInspector

    dataset: RerunInspectorDatasetConfig = Field(default_factory=RerunInspectorDatasetConfig)
    """Dataset reader settings."""

    selection: RerunInspectorSelectionConfig = Field(default_factory=RerunInspectorSelectionConfig)
    """Sample selection settings."""

    output: RerunInspectorOutputConfig = Field(default_factory=RerunInspectorOutputConfig)
    """Rerun output settings."""

    geometry: RerunInspectorGeometryConfig = Field(default_factory=RerunInspectorGeometryConfig)
    """Geometry primitive settings."""

    performance: RerunInspectorPerformanceConfig = Field(default_factory=RerunInspectorPerformanceConfig)
    """Runtime performance settings."""

    candidate: RerunInspectorCandidateConfig = Field(default_factory=RerunInspectorCandidateConfig)
    """Candidate subset and selected-detail logging policy."""

    efm_voxels: RerunInspectorEfmVoxelConfig = Field(default_factory=RerunInspectorEfmVoxelConfig)
    """EFM voxel-field visualization settings."""

    ase_keyframes: RerunInspectorAseKeyframeConfig = Field(default_factory=RerunInspectorAseKeyframeConfig)
    """ASE camera keyframe visualization settings."""

    primitives: RerunInspectorPrimitivesConfig = Field(default_factory=RerunInspectorPrimitivesConfig)
    """Primitive toggles."""


__all__ = [
    "RerunInspectorDatasetConfig",
    "RerunInspectorGeometryConfig",
    "RerunInspectorAseKeyframeConfig",
    "RerunInspectorCandidateConfig",
    "RerunInspectorEfmVoxelConfig",
    "RerunInspectorOutputConfig",
    "RerunInspectorPerformanceConfig",
    "RerunInspectorPrimitivesConfig",
    "RerunInspectorSelectionConfig",
    "RerunOfflineInspectorConfig",
]

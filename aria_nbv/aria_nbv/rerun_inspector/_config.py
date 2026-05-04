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
            load_counterfactuals=False,
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

    log_gt_trajectory: bool = True
    """Log the snippet rig trajectory when available."""

    log_candidate_depths: bool = False
    """Log candidate depth diagnostics when available."""

    log_rgb_keyframes: bool = False
    """Log live-attached RGB keyframes when a raw EFM snippet is attached."""

    log_depth_keyframes: bool = False
    """Log live-attached depth keyframes when a raw EFM snippet is attached."""


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

    primitives: RerunInspectorPrimitivesConfig = Field(default_factory=RerunInspectorPrimitivesConfig)
    """Primitive toggles."""


__all__ = [
    "RerunInspectorDatasetConfig",
    "RerunInspectorGeometryConfig",
    "RerunInspectorOutputConfig",
    "RerunInspectorPerformanceConfig",
    "RerunInspectorPrimitivesConfig",
    "RerunInspectorSelectionConfig",
    "RerunOfflineInspectorConfig",
]

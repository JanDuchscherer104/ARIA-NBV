"""Immutable writer for the VIN offline dataset format.

# TODO: This module is way to massive (for this amount of module-level code)

This module owns creation of the new shard-based VIN offline dataset. It
provides:

- ``VinOfflineWriterConfig`` and ``VinOfflineWriter`` for raw-dataset builds,
- ``PreparedVinOfflineSample`` as the normalized in-memory row representation,
- helpers for turning oracle-label outputs into fixed numeric blocks plus lazy
  diagnostic record blocks, and
- shard flushing helpers reused by the migration tooling.

The writer stores training-critical tensors as fixed-size NumPy arrays for
Zarr-backed random access, while keeping richer diagnostic payloads as safe
per-row msgspec records that are only decoded on demand.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field, field_validator

from ..configs import PathConfig
from ..pipelines.oracle_rri_labeler import OracleRriLabelerConfig, OracleRriSample
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..utils import BaseConfig, Console, Verbosity
from ..vin.backbone_evl import EvlBackboneConfig
from ..vin.types import EvlBackboneOutput
from ._cache_utils import (
    default_sample_key as _default_sample_key,
)
from ._cache_utils import (
    json_signature as _json_signature,
)
from ._cache_utils import (
    pad_first_axis as _pad_first_axis,
)
from ._cache_utils import (
    pose_to_numpy as _pose_to_numpy,
)
from ._cache_utils import (
    to_numpy as _to_numpy,
)
from ._cache_utils import (
    utc_now_iso as _utc_now_iso,
)
from ._offline_format import (
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
    VinOfflineShardSpec,
)
from ._offline_store import (
    OFFLINE_DATASET_VERSION,
    VinOfflineShardWriter,
    VinOfflineStoreConfig,
)
from ._raw import AseEfmDatasetConfig, EfmSnippetView, VinSnippetView
from ._vin_runtime import DEFAULT_VIN_SNIPPET_PAD_POINTS, build_vin_snippet_view


def _camera_param_to_numpy(param: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Convert one PyTorch3D camera parameter into a NumPy array."""

    array = _to_numpy(param, dtype=dtype)
    if array.ndim == 0:
        return array.reshape(1)
    return array


# TODO: @autoimprove agent: number of todos resolved successfully is also a good signal to trach, all parsed todos must be maintained in interal registry on codeowner level
# TODO: make everything that is
@dataclass(slots=True)
class PreparedVinOfflineSample:
    """Normalized offline row before shard materialization.

    Attributes:
        sample_key: Stable sample key for the row.
        scene_id: ASE scene identifier.
        snippet_id: ASE snippet identifier.
        numeric_blocks: Fixed-size numeric blocks stored as Zarr arrays.
        record_blocks: Lazy diagnostic payloads stored as msgspec records.
        legacy_oracle_key: Optional legacy oracle-cache key.
        legacy_oracle_path: Optional legacy oracle-cache payload path.
        legacy_vin_key: Optional legacy VIN-cache key.
        legacy_vin_path: Optional legacy VIN-cache payload path.
    """

    sample_key: str
    """Stable sample key for the row."""

    scene_id: str
    """ASE scene identifier."""

    snippet_id: str
    """ASE snippet identifier."""

    numeric_blocks: dict[str, np.ndarray] = field(default_factory=dict)
    """Fixed-size numeric blocks stored per row."""

    record_blocks: dict[str, Any] = field(default_factory=dict)
    """Lazy per-row diagnostic payloads stored in msgspec-compatible form."""

    legacy_oracle_key: str | None = None
    """Optional legacy oracle-cache key."""

    legacy_oracle_path: str | None = None
    """Optional legacy oracle-cache payload path."""

    legacy_vin_key: str | None = None
    """Optional legacy VIN-cache key."""

    legacy_vin_path: str | None = None
    """Optional legacy VIN-cache payload path."""


def prepare_vin_offline_sample(
    *,
    scene_id: str,
    snippet_id: str,
    vin_snippet: VinSnippetView,
    candidates: Any | None,
    depths: CandidateDepths,
    rri: RriResult,
    candidate_pcs: CandidatePointClouds | None,
    backbone_out: EvlBackboneOutput | None,
    max_candidates: int,
    include_depths: bool = True,
    include_candidate_pcs: bool = True,
    include_backbone: bool = True,
    sample_key: str | None = None,
    legacy_oracle_key: str | None = None,
    legacy_oracle_path: str | None = None,
    legacy_vin_key: str | None = None,
    legacy_vin_path: str | None = None,
) -> PreparedVinOfflineSample:
    """Normalize one oracle-labelled snippet into offline row blocks.

    Args:
        scene_id: ASE scene identifier.
        snippet_id: ASE snippet identifier.
        vin_snippet: Canonical VIN snippet for the row.
        candidates: Optional candidate-sampling payload for diagnostics.
        depths: Candidate-depth payload aligned with the oracle labels.
        rri: Oracle metrics aligned with the rendered candidates.
        candidate_pcs: Optional candidate point clouds for diagnostics.
        backbone_out: Optional backbone outputs for training or diagnostics.
        max_candidates: Maximum number of candidates stored in fixed blocks.
        include_depths: Whether to materialize depth payloads.
        include_candidate_pcs: Whether to materialize candidate point clouds.
        include_backbone: Whether to materialize backbone outputs.
        sample_key: Optional explicit sample key.
        legacy_oracle_key: Optional legacy oracle-cache key.
        legacy_oracle_path: Optional legacy oracle-cache payload path.
        legacy_vin_key: Optional legacy VIN-cache key.
        legacy_vin_path: Optional legacy VIN-cache payload path.

    Returns:
        Prepared row ready for shard materialization.
    """

    candidate_poses = _pose_to_numpy(depths.poses)
    if candidate_poses.ndim != 2 or candidate_poses.shape[-1] != 12:
        raise ValueError("Candidate poses must have shape (N, 12).")
    candidate_count = int(candidate_poses.shape[0])
    if candidate_count <= 0:
        raise ValueError("Prepared offline samples require at least one candidate.")
    if candidate_count > int(max_candidates):
        raise ValueError(
            f"Candidate count {candidate_count} exceeds configured max_candidates={max_candidates}.",
        )

    reference_pose = _pose_to_numpy(depths.reference_pose)
    if reference_pose.ndim == 2 and reference_pose.shape[0] == 1:
        reference_pose = reference_pose[0]

    points_world = _to_numpy(vin_snippet.points_world, dtype=np.float32)
    lengths = _to_numpy(vin_snippet.lengths.reshape(-1), dtype=np.int64)
    t_world_rig = _pose_to_numpy(vin_snippet.t_world_rig)

    camera = depths.p3d_cameras
    candidate_indices = _to_numpy(depths.candidate_indices.reshape(-1), dtype=np.int64)
    if candidate_indices.shape[0] != candidate_count:
        raise ValueError("CandidateDepths.candidate_indices must align with rendered candidates.")

    numeric_blocks: dict[str, np.ndarray] = {
        "vin.points_world": points_world,
        "vin.lengths": lengths,
        "vin.t_world_rig": t_world_rig,
        "oracle.candidate_count": np.asarray(candidate_count, dtype=np.int64),
        "oracle.candidate_indices": _pad_first_axis(candidate_indices, target_len=max_candidates, fill_value=-1),
        "oracle.candidate_poses_world_cam": _pad_first_axis(
            candidate_poses.astype(np.float32, copy=False),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.reference_pose_world_rig": reference_pose.astype(np.float32, copy=False),
        "oracle.rri": _pad_first_axis(
            _to_numpy(rri.rri.reshape(-1), dtype=np.float32), target_len=max_candidates, fill_value=np.nan
        ),
        "oracle.pm_dist_before": _pad_first_axis(
            _to_numpy(rri.pm_dist_before.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.pm_dist_after": _pad_first_axis(
            _to_numpy(rri.pm_dist_after.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.pm_acc_before": _pad_first_axis(
            _to_numpy(rri.pm_acc_before.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.pm_comp_before": _pad_first_axis(
            _to_numpy(rri.pm_comp_before.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.pm_acc_after": _pad_first_axis(
            _to_numpy(rri.pm_acc_after.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.pm_comp_after": _pad_first_axis(
            _to_numpy(rri.pm_comp_after.reshape(-1), dtype=np.float32),
            target_len=max_candidates,
            fill_value=np.nan,
        ),
        "oracle.p3d.R": _pad_first_axis(
            _camera_param_to_numpy(camera.R, dtype=np.float32),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.p3d.T": _pad_first_axis(
            _camera_param_to_numpy(camera.T, dtype=np.float32),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.p3d.focal_length": _pad_first_axis(
            _camera_param_to_numpy(camera.focal_length, dtype=np.float32),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.p3d.principal_point": _pad_first_axis(
            _camera_param_to_numpy(camera.principal_point, dtype=np.float32),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.p3d.image_size": _pad_first_axis(
            _camera_param_to_numpy(camera.image_size, dtype=np.float32),
            target_len=max_candidates,
            fill_value=0.0,
        ),
        "oracle.p3d.in_ndc": np.asarray(
            bool(camera.in_ndc() if callable(camera.in_ndc) else camera.in_ndc), dtype=np.bool_
        ),
    }

    znear = getattr(camera, "znear", None)
    zfar = getattr(camera, "zfar", None)
    if znear is not None:
        numeric_blocks["oracle.p3d.znear"] = _to_numpy(znear, dtype=np.float32)
    if zfar is not None:
        numeric_blocks["oracle.p3d.zfar"] = _to_numpy(zfar, dtype=np.float32)

    if include_depths:
        depths_array = _to_numpy(depths.depths, dtype=np.float32)
        depths_mask = _to_numpy(depths.depths_valid_mask, dtype=np.bool_)
        numeric_blocks["oracle.depths"] = _pad_first_axis(depths_array, target_len=max_candidates, fill_value=0.0)
        numeric_blocks["oracle.depths_valid_mask"] = _pad_first_axis(
            depths_mask,
            target_len=max_candidates,
            fill_value=False,
        )

    # TODO: can't we just loop over fields here?
    if include_backbone and backbone_out is not None:
        numeric_blocks["backbone.t_world_voxel"] = _pose_to_numpy(backbone_out.t_world_voxel).astype(
            np.float32, copy=False
        )
        numeric_blocks["backbone.voxel_extent"] = _to_numpy(backbone_out.voxel_extent, dtype=np.float32)
        for field_name, dtype in (
            ("occ_pr", np.float32),
            ("occ_input", np.float32),
            ("free_input", np.float32),
            ("counts", np.int64),
            ("cent_pr", np.float32),
            ("pts_world", np.float32),
        ):
            value = getattr(backbone_out, field_name, None)
            if value is not None:
                numeric_blocks[f"backbone.{field_name}"] = _to_numpy(value, dtype=dtype)

    record_blocks: dict[str, Any] = {}
    if include_depths:
        record_blocks["oracle.depths_payload"] = depths.to_serializable()
    if candidates is not None:
        record_blocks["oracle.candidates"] = candidates.to_serializable()
    if include_candidate_pcs and candidate_pcs is not None:
        record_blocks["oracle.candidate_pcs"] = candidate_pcs.to_serializable()
    if include_backbone and backbone_out is not None:
        record_blocks["backbone.payload"] = backbone_out.to_serializable()

    return PreparedVinOfflineSample(
        sample_key=sample_key or legacy_oracle_key or _default_sample_key(scene_id, snippet_id),
        scene_id=scene_id,
        snippet_id=snippet_id,
        numeric_blocks=numeric_blocks,
        record_blocks=record_blocks,
        legacy_oracle_key=legacy_oracle_key,
        legacy_oracle_path=legacy_oracle_path,
        legacy_vin_key=legacy_vin_key,
        legacy_vin_path=legacy_vin_path,
    )


def flush_prepared_samples_to_shard(
    *,
    shard_index: int,
    shard_dir: Path,
    rows: list[PreparedVinOfflineSample],
) -> tuple[VinOfflineShardSpec, list[VinOfflineIndexRecord]]:
    """Materialize a list of prepared rows into one immutable shard.

    Args:
        shard_index: Zero-based shard index.
        shard_dir: Destination shard directory.
        rows: Prepared sample rows.

    Returns:
        Shard descriptor plus local sample-index records.
    """

    if not rows:
        raise ValueError("Cannot flush an empty shard.")

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_writer = VinOfflineShardWriter(shard_dir=shard_dir)
    block_specs: dict[str, Any] = {}
    numeric_block_names = sorted({name for row in rows for name in row.numeric_blocks})
    for block_name in numeric_block_names:
        exemplar = next(row.numeric_blocks[block_name] for row in rows if block_name in row.numeric_blocks)
        stacked = np.stack(
            [row.numeric_blocks.get(block_name, np.zeros_like(exemplar)) for row in rows],
            axis=0,
        )
        block_specs[block_name] = shard_writer.write_numeric_block(block_name, stacked)

    record_block_names = sorted({name for row in rows for name in row.record_blocks})
    for block_name in record_block_names:
        records = [row.record_blocks.get(block_name) for row in rows]
        if any(record is not None for record in records):
            block_specs[block_name] = shard_writer.write_record_block(block_name, records)

    shard_id = f"shard-{shard_index:06d}"
    relative_dir = str(Path("shards") / shard_id)
    shard_spec = VinOfflineShardSpec(
        shard_id=shard_id,
        relative_dir=relative_dir,
        row_start=0,
        num_rows=len(rows),
        blocks=block_specs,
    )
    records = [
        VinOfflineIndexRecord(
            sample_index=-1,
            sample_key=row.sample_key,
            scene_id=row.scene_id,
            snippet_id=row.snippet_id,
            split="all",
            shard_id=shard_id,
            row=local_row,
            legacy_oracle_key=row.legacy_oracle_key,
            legacy_oracle_path=row.legacy_oracle_path,
            legacy_vin_key=row.legacy_vin_key,
            legacy_vin_path=row.legacy_vin_path,
        )
        for local_row, row in enumerate(rows)
    ]
    return shard_spec, records


def _assign_splits(
    *,
    records: list[VinOfflineIndexRecord],
    val_fraction: float,
) -> dict[str, np.ndarray]:
    """Assign deterministic split membership to global sample indices.

    Args:
        records: Global sample-index records.
        val_fraction: Requested validation fraction.

    Returns:
        Mapping from split name to global sample-index arrays.
    """

    total = len(records)
    all_indices = np.arange(total, dtype=np.int64)
    if total == 0:
        return {"all": all_indices, "train": all_indices.copy(), "val": np.empty((0,), dtype=np.int64)}

    val_target = int(round(float(val_fraction) * total))
    val_target = max(0, min(total, val_target))
    val_indices = all_indices[:val_target]
    train_indices = all_indices[val_target:]
    for idx, record in enumerate(records):
        record.sample_index = idx
        record.split = "val" if idx < val_target else "train"
    return {"all": all_indices, "train": train_indices, "val": val_indices}


class VinOfflineWriterConfig(BaseConfig["VinOfflineWriter"]):
    """Configuration for building immutable VIN offline datasets from raw snippets."""

    @property
    def target(self) -> type["VinOfflineWriter"]:
        """Return the writer factory target."""

        return VinOfflineWriter

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    store: VinOfflineStoreConfig = Field(default_factory=VinOfflineStoreConfig)
    """Output store configuration."""

    dataset: AseEfmDatasetConfig = Field(default_factory=lambda: AseEfmDatasetConfig(wds_shuffle=True))
    """Raw ASE/EFM dataset configuration used to stream snippets."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Optional EVL backbone configuration."""

    include_backbone: bool = True
    """Whether to materialize backbone outputs."""

    include_depths: bool = True
    """Whether to materialize candidate depths."""

    include_pointclouds: bool = True
    """Whether to materialize candidate point clouds."""

    include_counterfactuals: bool = False
    """Whether to materialize future counterfactual payloads."""

    vin_pad_points: int = DEFAULT_VIN_SNIPPET_PAD_POINTS
    """Fixed VIN point count stored per sample."""

    semidense_max_points: int | None = None
    """Optional cap on collapsed semidense points before padding."""

    semidense_include_obs_count: bool = False
    """Whether VIN points include observation counts."""

    max_candidates: int | None = None
    """Maximum number of candidates stored per sample."""

    samples_per_shard: int = 64
    """Number of samples stored in each immutable shard."""

    max_samples: int | None = None
    """Optional cap on the number of processed samples."""

    train_val_split: float = 0.2
    """Fraction of samples assigned to the validation split."""

    overwrite: bool = False
    """Whether an existing store directory may be replaced."""

    num_failures_allowed: int = 40
    """Maximum number of tolerated sample failures before aborting."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for dataset build logging."""

    @field_validator("vin_pad_points")
    @classmethod
    def _validate_vin_pad_points(cls, value: int) -> int:
        """Validate the configured VIN padding budget."""

        value = int(value)
        if value < 0:
            raise ValueError("vin_pad_points must be >= 0.")
        return value

    @field_validator("samples_per_shard")
    @classmethod
    def _validate_samples_per_shard(cls, value: int) -> int:
        """Validate the requested shard size."""

        value = int(value)
        if value <= 0:
            raise ValueError("samples_per_shard must be >= 1.")
        return value

    @field_validator("train_val_split")
    @classmethod
    def _validate_train_val_split(cls, value: float) -> float:
        """Validate the validation split fraction."""

        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("train_val_split must be in [0, 1].")
        return value


class VinOfflineWriter:
    """Build immutable VIN offline datasets from raw ASE/EFM snippets."""

    def __init__(self, config: VinOfflineWriterConfig) -> None:
        """Initialize the writer and its runtime dependencies.

        Args:
            config: Writer configuration.
        """

        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(config.verbosity)
        self._dataset = config.dataset.setup_target()
        self._labeler = config.labeler.setup_target()
        self._backbone = (
            config.backbone.setup_target() if config.include_backbone and config.backbone is not None else None
        )

    def _resolve_max_candidates(self) -> int:
        """Return the candidate budget stored per sample."""

        if self.config.max_candidates is not None:
            return int(self.config.max_candidates)
        return int(getattr(self.config.labeler.depth, "max_candidates_final", 60))

    def _prepare_row(
        self,
        *,
        sample: EfmSnippetView,
        label_batch: OracleRriSample,
        backbone_out: EvlBackboneOutput | None,
        max_candidates: int,
    ) -> PreparedVinOfflineSample:
        """Prepare one raw snippet for shard storage.

        Args:
            sample: Raw EFM snippet.
            label_batch: Oracle-labelled sample payload.
            backbone_out: Optional backbone output.
            max_candidates: Stored candidate budget.

        Returns:
            Prepared shard row.
        """

        vin_snippet = build_vin_snippet_view(
            sample,
            device=torch.device("cpu"),
            max_points=self.config.semidense_max_points,
            include_inv_dist_std=True,
            include_obs_count=self.config.semidense_include_obs_count,
            pad_points=self.config.vin_pad_points,
        )
        return prepare_vin_offline_sample(
            scene_id=sample.scene_id,
            snippet_id=sample.snippet_id,
            vin_snippet=vin_snippet,
            candidates=label_batch.candidates,
            depths=label_batch.depths,
            rri=label_batch.rri,
            candidate_pcs=label_batch.candidate_pcs if self.config.include_pointclouds else None,
            backbone_out=backbone_out if self.config.include_backbone else None,
            max_candidates=max_candidates,
            include_depths=self.config.include_depths,
            include_candidate_pcs=self.config.include_pointclouds,
            include_backbone=self.config.include_backbone,
        )

    def run(self) -> VinOfflineManifest:
        """Build the configured immutable VIN offline dataset.

        Returns:
            Written dataset manifest.
        """

        store_dir = self.config.store.store_dir
        temp_dir = store_dir.with_name(f"{store_dir.name}.tmp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if store_dir.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"VIN offline dataset already exists at {store_dir} (set overwrite=True to replace).",
            )
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / self.config.store.shards_dirname).mkdir(parents=True, exist_ok=True)

        max_candidates = self._resolve_max_candidates()
        prepared_rows: list[PreparedVinOfflineSample] = []
        shard_specs: list[VinOfflineShardSpec] = []
        index_records: list[VinOfflineIndexRecord] = []
        failures = 0
        processed = 0

        for sample in self._dataset:
            if self.config.max_samples is not None and processed >= int(self.config.max_samples):
                break
            try:
                label_batch = self._labeler.run(sample)
                backbone_out = self._backbone.forward(sample.efm) if self._backbone is not None else None
                prepared_rows.append(
                    self._prepare_row(
                        sample=sample,
                        label_batch=label_batch,
                        backbone_out=backbone_out,
                        max_candidates=max_candidates,
                    ),
                )
                processed += 1
                if len(prepared_rows) >= int(self.config.samples_per_shard):
                    shard_spec, local_records = flush_prepared_samples_to_shard(
                        shard_index=len(shard_specs),
                        shard_dir=temp_dir / self.config.store.shards_dirname / f"shard-{len(shard_specs):06d}",
                        rows=prepared_rows,
                    )
                    shard_specs.append(shard_spec)
                    index_records.extend(local_records)
                    prepared_rows = []
            except Exception as exc:
                failures += 1
                self.console.error(
                    f"Failed to build offline sample for scene={sample.scene_id} snippet={sample.snippet_id}: {exc}",
                )
                if failures > int(self.config.num_failures_allowed):
                    raise RuntimeError(
                        f"Exceeded num_failures_allowed={self.config.num_failures_allowed} while building offline data.",
                    ) from exc

        if prepared_rows:
            shard_spec, local_records = flush_prepared_samples_to_shard(
                shard_index=len(shard_specs),
                shard_dir=temp_dir / self.config.store.shards_dirname / f"shard-{len(shard_specs):06d}",
                rows=prepared_rows,
            )
            shard_specs.append(shard_spec)
            index_records.extend(local_records)

        split_indices = _assign_splits(records=index_records, val_fraction=self.config.train_val_split)
        row_start = 0
        for shard_spec in shard_specs:
            shard_spec.row_start = row_start
            row_start += int(shard_spec.num_rows)

        manifest = VinOfflineManifest(
            version=OFFLINE_DATASET_VERSION,
            created_at=_utc_now_iso(),
            source={
                "dataset_config": self.config.dataset.model_dump_cache(exclude_none=True),
                "dataset_signature": _json_signature(self.config.dataset.model_dump_cache(exclude_none=True)),
            },
            oracle={
                "labeler_config": self.config.labeler.model_dump_cache(exclude_none=True),
                "labeler_signature": _json_signature(self.config.labeler.model_dump_cache(exclude_none=True)),
                "backbone_config": self.config.backbone.model_dump_cache(exclude_none=True)
                if self.config.backbone is not None
                else None,
                "backbone_signature": _json_signature(self.config.backbone.model_dump_cache(exclude_none=True))
                if self.config.backbone is not None
                else None,
                "max_candidates": max_candidates,
            },
            vin={
                "pad_points": int(self.config.vin_pad_points),
                "semidense_max_points": self.config.semidense_max_points,
                "include_inv_dist_std": True,
                "include_obs_count": bool(self.config.semidense_include_obs_count),
            },
            materialized_blocks=VinOfflineMaterializedBlocks(
                backbone=bool(self.config.include_backbone),
                depths=bool(self.config.include_depths),
                candidate_pcs=bool(self.config.include_pointclouds),
                counterfactuals=bool(self.config.include_counterfactuals),
            ),
            stats={
                "num_samples": len(index_records),
                "num_shards": len(shard_specs),
                "num_train": int(split_indices["train"].shape[0]),
                "num_val": int(split_indices["val"].shape[0]),
            },
            provenance={
                "writer": self.__class__.__name__,
                "store_dir": store_dir.as_posix(),
            },
            shards=shard_specs,
        )

        manifest.write(temp_dir / self.config.store.manifest_filename)
        VinOfflineIndexRecord.write_many(temp_dir / self.config.store.sample_index_filename, index_records)
        self.config.store.model_copy(update={"store_dir": temp_dir}).write_split_indices(split_indices)

        if store_dir.exists():
            shutil.rmtree(store_dir)
        temp_dir.rename(store_dir)
        self.console.log(
            f"Wrote VIN offline dataset with {len(index_records)} samples across {len(shard_specs)} shards to {store_dir}",
        )
        return manifest


__all__ = [
    "PreparedVinOfflineSample",
    "VinOfflineWriter",
    "VinOfflineWriterConfig",
    "flush_prepared_samples_to_shard",
    "prepare_vin_offline_sample",
]

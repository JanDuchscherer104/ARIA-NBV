"""Oracle cache v2 writer, config, and index-management helpers.

This module contains the non-dataset pieces of the v2 offline oracle-cache
surface. It owns:
- filesystem config models for oracle-cache directories,
- writer logic for serializing oracle labels and optional backbone outputs,
- split-index repair and rebuild helpers,
- shared payload encoding/decoding helpers consumed by the dataset module, and
- re-exports for the dataset classes defined in `oracle_cache_datasets.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, ValidationInfo, field_validator

from ..configs import PathConfig
from ..pipelines.oracle_rri_labeler import OracleRriLabelerConfig, OracleRriSample
from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rri_metrics.types import RriResult
from ..utils import BaseConfig, Console, Verbosity
from ..vin.backbone_evl import EvlBackboneConfig
from ..vin.types import EvlBackboneOutput
from ._cache_utils import format_sample_key, unique_sample_path
from .cache_contracts import (
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
)
from .cache_index import (
    read_index,
    read_pairs,
    rebuild_oracle_entries_from_samples,
    repair_oracle_split_indices,
    write_index,
)
from .efm_dataset import AseEfmDatasetConfig
from .efm_views import EfmSnippetView
from .offline_cache_store import (
    _read_metadata,
    _write_metadata,
    build_cache_metadata,
    snapshot_config,
    snapshot_dataset_config,
)
from .vin_cache import VinSnippetCacheConfig

if TYPE_CHECKING:
    from .oracle_cache_datasets import OracleRriCacheDataset, OracleRriCacheVinDataset

CACHE_VERSION = 1

decode_candidates = CandidateSamplingResult.from_serializable
decode_depths = CandidateDepths.from_serializable
decode_candidate_pcs = CandidatePointClouds.from_serializable
decode_rri = RriResult.from_serializable
decode_backbone = EvlBackboneOutput.from_serializable


class OracleRriCacheConfig(BaseConfig):
    """Filesystem configuration for oracle cache artifacts."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache_dir: Path = Field(default_factory=lambda: PathConfig().offline_cache_dir)
    """Root directory containing cached samples."""

    index_filename: str = "index.jsonl"
    """Filename for the cache index."""

    metadata_filename: str = "metadata.json"
    """Filename for the cache metadata."""

    samples_dirname: str = "samples"
    """Subdirectory name that stores sample payloads."""

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, value: str | Path, info: ValidationInfo) -> Path:
        """Resolve relative cache directories against the configured data roots."""
        paths: PathConfig = info.data.get("paths") or PathConfig()
        return paths.resolve_cache_dir(value)

    @property
    def samples_dir(self) -> Path:
        """Return the directory containing cached sample payloads."""
        return self.cache_dir / self.samples_dirname

    @property
    def index_path(self) -> Path:
        """Return the path to the base oracle-cache JSONL index."""
        return self.cache_dir / self.index_filename

    @property
    def metadata_path(self) -> Path:
        """Return the path to the oracle-cache metadata JSON."""
        return self.cache_dir / self.metadata_filename

    @property
    def train_index_path(self) -> Path:
        """Return the path to the train split JSONL index."""
        return self.cache_dir / "train_index.jsonl"

    @property
    def val_index_path(self) -> Path:
        """Return the path to the validation split JSONL index."""
        return self.cache_dir / "val_index.jsonl"

    def rebuild_index(
        self,
        *,
        train_val_split: float | None = None,
        rng_seed: int | None = None,
    ) -> int:
        """Rebuild base and split indices from cached sample filenames.

        This repair path scans the cache sample directory, regenerates the base
        oracle-cache index, refreshes the train/validation split indices, and
        updates the stored metadata count when metadata already exists.

        Args:
            train_val_split: Optional validation fraction override. When
                omitted, the default split from
                `OracleRriCacheDatasetConfig.train_val_split` is used.
            rng_seed: Optional deterministic seed applied when shuffling
                samples before the train/validation split is rebuilt.

        Returns:
            The number of sample entries written to the rebuilt base index.
        """
        split = (
            float(train_val_split)
            if train_val_split is not None
            else float(OracleRriCacheDatasetConfig().train_val_split)
        )
        samples_dir = self.samples_dir
        if not samples_dir.exists():
            write_index(self.index_path, [])
            repair_oracle_cache_indices(
                cache=self,
                train_val_split=split,
            )
            return 0

        entries = rebuild_oracle_entries_from_samples(
            samples_dir=samples_dir,
            cache_dir=self.cache_dir,
            rng_seed=rng_seed,
        )
        write_index(self.index_path, entries)
        repair_oracle_cache_indices(
            cache=self,
            train_val_split=split,
        )
        meta_path = self.metadata_path
        if meta_path.exists():
            meta = _read_metadata(meta_path)
            meta.num_samples = len(entries)
            _write_metadata(meta_path, meta)
        return len(entries)


class OracleRriCacheWriterConfig(BaseConfig["OracleRriCacheWriter"]):
    """Configuration for building oracle caches from raw ASE snippets."""

    @property
    def target(self) -> type["OracleRriCacheWriter"]:
        """Return the writer factory target."""
        return OracleRriCacheWriter

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheConfig = Field(default_factory=OracleRriCacheConfig)
    """Cache path configuration."""

    dataset: AseEfmDatasetConfig = Field(default_factory=lambda: AseEfmDatasetConfig(wds_shuffle=True))
    """Dataset configuration used to stream snippets."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """Optional EVL backbone configuration."""

    num_failures_allowed: int = 40
    """Maximum number of allowed sample processing failures before aborting."""

    max_samples: int | None = None
    """Optional cap on number of cached samples."""

    include_backbone: bool = True
    """Whether to cache EVL backbone outputs."""

    include_depths: bool = True
    """Whether to cache candidate depth maps."""

    include_pointclouds: bool = True
    """Whether to cache candidate point clouds."""

    overwrite: bool = False
    """Allow overwriting an existing cache directory."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""


class OracleRriCacheDatasetConfig(BaseConfig["OracleRriCacheDataset"]):
    """Configuration for loading cached oracle outputs."""

    @property
    def target(self) -> type["OracleRriCacheDataset"]:
        """Return the dataset factory target."""
        from .oracle_cache_datasets import OracleRriCacheDataset

        return OracleRriCacheDataset

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheConfig = Field(default_factory=OracleRriCacheConfig)
    """Cache path configuration."""

    load_backbone: bool = True
    """Whether to load cached EVL backbone outputs."""

    backbone_keep_fields: list[str] | None = None
    """Optional allowlist of EVL backbone fields to decode."""

    include_efm_snippet: bool = True
    """Whether to include a snippet view in loaded samples."""

    include_gt_mesh: bool = False
    """Whether to include the ground truth mesh when loading EFM snippets."""

    load_candidates: bool = True
    """Whether to decode candidate sampling metadata."""

    load_depths: bool = True
    """Whether to decode candidate depth maps."""

    load_candidate_pcs: bool = True
    """Whether to decode candidate point clouds."""

    efm_keep_keys: list[str] | None = None
    """Optional allowlist of EFM keys to keep when loading snippets."""

    vin_snippet_cache: VinSnippetCacheConfig | None = None
    """Optional precomputed cache for minimal VIN snippet entries."""

    semidense_max_points: int | None = None
    """Optional cap on the number of collapsed semidense points."""

    semidense_include_obs_count: bool = False
    """Whether to append semidense observation counts."""

    vin_snippet_cache_mode: Literal["auto", "required", "disabled"] = "auto"
    """Whether vin_snippet_cache is required, optional, or disabled."""

    vin_snippet_cache_allow_subset: bool = False
    """Whether to filter cache entries to those present in vin_snippet_cache."""

    return_format: Literal["cache_sample", "vin_batch"] = "cache_sample"
    """Return cached samples or VIN-ready batches."""

    simplification: float | None = None
    """Optional fraction of cached samples to expose."""

    train_val_split: float = 0.2
    """Fraction of cached samples to reserve for validation when splitting."""

    split: Literal["all", "train", "val"] = "all"
    """Subset of cached entries to load."""

    limit: int | None = None
    """Optional cap on number of cached samples loaded."""

    @field_validator("train_val_split", mode="before")
    @classmethod
    def _validate_train_val_split(cls, value: float) -> float:
        """Validate that the requested split fraction is in ``[0, 1]``."""
        split = float(value)
        if not 0.0 <= split <= 1.0:
            raise ValueError("train_val_split must be in [0, 1].")
        return split

    @field_validator("simplification", mode="before")
    @classmethod
    def _validate_simplification(cls, value: float | None) -> float | None:
        """Validate that simplification fractions are in ``(0, 1)``."""
        if value is None:
            return None
        simplification = float(value)
        if not 0.0 < simplification < 1.0:
            raise ValueError("simplification must be in (0, 1).")
        return simplification


def _read_vin_snippet_cache_index(path: Path, *, allow_missing: bool = False) -> set[tuple[str, str]]:
    """Read available ``(scene_id, snippet_id)`` pairs from a VIN cache index."""
    return read_pairs(path, allow_missing=allow_missing)


def build_cache_payload(
    label_batch: OracleRriSample,
    *,
    backbone_out: EvlBackboneOutput | None,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
) -> dict[str, Any]:
    """Serialize a label batch and optional backbone output."""
    payload: dict[str, Any] = {
        "scene_id": label_batch.sample.scene_id,
        "snippet_id": label_batch.sample.snippet_id,
        "candidates": label_batch.candidates.to_serializable(),
        "depths": label_batch.depths.to_serializable() if include_depths else None,
        "candidate_pcs": label_batch.candidate_pcs.to_serializable() if include_pointclouds else None,
        "rri": label_batch.rri.to_serializable(),
    }
    if include_backbone:
        if backbone_out is None:
            raise ValueError("include_backbone=True requires backbone_out to be provided.")
        payload["backbone"] = backbone_out.to_serializable()
    return payload


def repair_oracle_cache_indices(
    *,
    cache: OracleRriCacheConfig,
    train_val_split: float,
    console: Console | None = None,
) -> tuple[list[OracleRriCacheEntry], list[OracleRriCacheEntry]]:
    """Repair oracle train/val split indices from the base index."""
    base_entries = read_index(cache.index_path, entry_type=OracleRriCacheEntry, allow_missing=True)
    return repair_oracle_split_indices(
        base_entries=base_entries,
        train_index_path=cache.train_index_path,
        val_index_path=cache.val_index_path,
        val_fraction=train_val_split,
        console=console,
    )


class OracleRriCacheWriter:
    """Build an offline cache of oracle labels and optional EVL outputs."""

    def __init__(self, config: OracleRriCacheWriterConfig) -> None:
        """Initialize the oracle-cache writer and its runtime dependencies."""
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(self.config.verbosity)
        self.console.log(f"Writing to cache directory: {self.config.cache.cache_dir}")

        self._dataset = self.config.dataset.setup_target()
        self._labeler = self.config.labeler.setup_target()
        self._backbone = None
        if self.config.include_backbone:
            if self.config.backbone is None:
                raise ValueError("include_backbone=True requires backbone config.")
            self._backbone = self.config.backbone.setup_target()
        self._config_hash: str | None = None

    def run(self) -> list[OracleRriCacheEntry]:
        """Materialize the configured dataset into the oracle-cache directory.

        The writer prepares the cache directory structure and metadata, then
        iterates the configured dataset to serialize oracle labels and optional
        backbone outputs. When an index already exists, `overwrite=True` puts
        the method into resume mode: retained entries are validated against
        their sample payloads, missing payloads are dropped from the in-memory
        index, and duplicate `(scene_id, snippet_id)` pairs are skipped so only
        uncached samples are recomputed. The `max_samples` limit is applied to
        the total cache size after any retained entries are counted.

        Individual sample failures are logged and tolerated until the configured
        failure budget is exceeded. Once iteration finishes, the method rewrites
        the base index, repairs the derived train/validation split indices, and
        refreshes metadata with the final sample count before returning only the
        entries created during the current invocation.

        Returns:
            The cache entries written during this invocation, excluding any
            retained entries loaded from a pre-existing index.

        Raises:
            FileExistsError: If the cache index already exists and
                `overwrite=False`.
            RuntimeError: If the number of failed samples exceeds
                `config.num_failures_allowed`.
        """
        cache_dir = self.config.cache.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = self.config.cache.samples_dir
        samples_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.config.cache.index_path
        existing_entries: list[OracleRriCacheEntry] = []
        existing_pairs: set[tuple[str, str]] = set()
        if index_path.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Cache index already exists at {index_path} (set overwrite=True to replace).",
                )
            existing_entries = read_index(index_path, entry_type=OracleRriCacheEntry, allow_missing=True)
            valid_entries: list[OracleRriCacheEntry] = []
            for entry in existing_entries:
                entry_path = self.config.cache.cache_dir / entry.path
                if entry_path.exists():
                    valid_entries.append(entry)
                    existing_pairs.add((entry.scene_id, entry.snippet_id))
                else:
                    self.console.warn(
                        "Index entry missing sample file; it will be regenerated if encountered "
                        f"(scene={entry.scene_id} snippet={entry.snippet_id}).",
                    )
            existing_entries = valid_entries
            if existing_entries:
                self.console.log(
                    f"Found {len(existing_entries)} existing cached samples; skipping duplicates by scene/snippet.",
                )

        meta = build_cache_metadata(
            labeler=self.config.labeler,
            dataset=self.config.dataset,
            backbone=self.config.backbone,
            include_backbone=self.config.include_backbone,
            include_depths=self.config.include_depths,
            include_pointclouds=self.config.include_pointclouds,
            version=CACHE_VERSION,
        )
        _write_metadata(self.config.cache.metadata_path, meta)

        self._config_hash = meta.config_hash
        entries: list[OracleRriCacheEntry] = []
        max_samples = self.config.max_samples
        count = len(existing_entries)
        skipped = 0
        fail_count = 0

        for sample in self._dataset:
            try:
                if max_samples is not None and count >= max_samples:
                    break
                sample_pair = (sample.scene_id, sample.snippet_id)
                if sample_pair in existing_pairs:
                    skipped += 1
                    continue
                cache_entry = self._write_sample(sample, samples_dir)
                entries.append(cache_entry)
                count += 1
                existing_pairs.add(sample_pair)
                if max_samples is not None and count >= max_samples:
                    break
            except Exception as exc:
                self.console.error(
                    f"Failed to cache scene={sample.scene_id} snippet={sample.snippet_id}: {exc}",
                )
                fail_count += 1
                continue
            finally:
                if fail_count > self.config.num_failures_allowed:
                    raise RuntimeError(
                        f"Exceeded maximum allowed failures ({self.config.num_failures_allowed}). Aborting cache write.",
                    )

        final_entries = [*existing_entries, *entries]
        write_index(index_path, final_entries)
        repair_oracle_cache_indices(
            cache=self.config.cache,
            train_val_split=OracleRriCacheDatasetConfig().train_val_split,
            console=self.console,
        )

        if skipped:
            self.console.log(f"Skipped {skipped} duplicate samples already in cache.")
        meta.num_samples = count
        _write_metadata(self.config.cache.metadata_path, meta)
        self.console.log(
            f"Cache now contains {count} samples (added {len(entries)} new) in {cache_dir}",
        )
        return entries

    def _write_sample(
        self,
        sample: EfmSnippetView,
        samples_dir: Path,
    ) -> OracleRriCacheEntry:
        """Run oracle labeling for one snippet and persist the encoded payload."""
        label_batch = self._labeler.run(sample)
        backbone_out = None
        if self._backbone is not None:
            backbone_out = self._backbone.forward(sample.efm)

        payload = self._encode_sample(label_batch, backbone_out=backbone_out)
        config_hash = self._config_hash or "unknown"
        base_key = format_sample_key(sample.scene_id, sample.snippet_id, config_hash)
        sample_key, sample_path = unique_sample_path(samples_dir, base_key)
        torch.save(payload, sample_path)

        self.console.log(f"Cached scene={sample.scene_id} snippet={sample.snippet_id} -> {sample_path.name}")
        return OracleRriCacheEntry(
            key=sample_key,
            scene_id=sample.scene_id,
            snippet_id=sample.snippet_id,
            path=str(sample_path.relative_to(self.config.cache.cache_dir)),
        )

    def _encode_sample(
        self,
        label_batch: OracleRriSample,
        *,
        backbone_out: EvlBackboneOutput | None,
    ) -> dict[str, Any]:
        """Encode one oracle-label batch into its on-disk cache payload."""
        return build_cache_payload(
            label_batch,
            backbone_out=backbone_out,
            include_backbone=self.config.include_backbone,
            include_depths=self.config.include_depths,
            include_pointclouds=self.config.include_pointclouds,
        )


def __getattr__(name: str) -> Any:
    """Lazily expose dataset classes that live in `oracle_cache_datasets.py`."""
    if name in {"OracleRriCacheDataset", "OracleRriCacheVinDataset"}:
        from .oracle_cache_datasets import OracleRriCacheDataset, OracleRriCacheVinDataset

        return {
            "OracleRriCacheDataset": OracleRriCacheDataset,
            "OracleRriCacheVinDataset": OracleRriCacheVinDataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CACHE_VERSION",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheVinDataset",
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "build_cache_payload",
    "repair_oracle_cache_indices",
    "snapshot_config",
    "snapshot_dataset_config",
]

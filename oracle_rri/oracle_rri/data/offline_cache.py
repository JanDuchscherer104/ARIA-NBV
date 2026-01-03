"""Offline cache utilities for oracle RRI labels and EVL backbone outputs.

This module provides:
1) snapshot helpers for configs (excluding huge path lists),
2) serialization helpers for oracle label outputs + EVL backbone outputs,
3) a cache writer that builds an offline dataset, and
4) a cache dataset for fast parallel reading.
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, ValidationInfo, field_validator
from torch.utils.data import Dataset, get_worker_info

from ..configs import PathConfig
from ..pipelines.oracle_rri_labeler import OracleRriLabelBatch, OracleRriLabelerConfig
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rri_metrics.types import RriResult
from ..utils import BaseConfig, Console, Verbosity
from ..vin.backbone_evl import EvlBackboneConfig
from ..vin.types import EvlBackboneOutput
from .efm_dataset import AseEfmDataset, AseEfmDatasetConfig
from .efm_views import EfmSnippetView
from .offline_cache_serialization import (
    decode_backbone,
    decode_candidate_pcs,
    decode_candidates,
    decode_depths,
    decode_rri,
    encode_backbone,
    encode_candidate_pcs,
    encode_candidates,
    encode_depths,
    encode_rri,
)
from .offline_cache_store import (
    _config_signature,
    _ensure_config_hash,
    _format_sample_key,
    _read_metadata,
    _unique_sample_path,
    _write_metadata,
    build_cache_metadata,
    snapshot_config,
    snapshot_dataset_config,
)
from .offline_cache_types import (
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
)

if TYPE_CHECKING:
    from ..lightning.lit_datamodule import VinOracleBatch

CACHE_VERSION = 1


# ----------------------------------------------------------------------------- Cache configs
def _target_cache_dataset() -> type[OracleRriCacheDataset]:
    return OracleRriCacheDataset


def _target_cache_writer() -> type[OracleRriCacheWriter]:
    return OracleRriCacheWriter


def _target_cache_appender() -> type[OracleRriCacheAppender]:
    return OracleRriCacheAppender


class OracleRriCacheConfig(BaseConfig):
    """Filesystem configuration for oracle cache artifacts."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache_dir: Path = Field(
        default_factory=lambda: PathConfig().offline_cache_dir,
    )
    """Root directory containing cached samples (resolved under PathConfig)."""

    index_filename: str = "index.jsonl"
    """Filename for the cache index (JSON lines)."""

    metadata_filename: str = "metadata.json"
    """Filename for the cache metadata JSON."""

    samples_dirname: str = "samples"
    """Subdirectory name that stores sample payloads."""

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, value: str | Path, info: ValidationInfo) -> Path:
        paths: PathConfig = info.data.get("paths") or PathConfig()
        path = Path(value)
        if path.is_absolute():
            return path.expanduser().resolve()
        base_dir = paths.offline_cache_dir or paths.data_root
        if path.parts:
            if path.parts[0] == paths.data_root.name or (
                paths.offline_cache_dir is not None and path.parts[0] == paths.offline_cache_dir.name
            ):
                base_dir = paths.root
        return paths.resolve_under_root(path, base_dir=base_dir)

    @property
    def samples_dir(self) -> Path:
        return self.cache_dir / self.samples_dirname

    @property
    def index_path(self) -> Path:
        return self.cache_dir / self.index_filename

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / self.metadata_filename

    @property
    def train_index_path(self) -> Path:
        return self.cache_dir / "train_index.jsonl"

    @property
    def val_index_path(self) -> Path:
        return self.cache_dir / "val_index.jsonl"


class OracleRriCacheWriterConfig(BaseConfig["OracleRriCacheWriter"]):
    """Configuration for building oracle caches from raw ASE snippets."""

    target: type[OracleRriCacheWriter] = Field(
        default_factory=_target_cache_writer,
        exclude=True,
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheConfig = Field(default_factory=OracleRriCacheConfig)
    """Cache path configuration."""

    dataset: AseEfmDatasetConfig = Field(
        default_factory=lambda: AseEfmDatasetConfig(wds_shuffle=True),
    )
    """Dataset configuration used to stream snippets (shuffled by default)."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """EVL backbone configuration (optional if include_backbone=False)."""

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


class OracleRriCacheAppenderConfig(BaseConfig["OracleRriCacheAppender"]):
    """Configuration for appending cached oracle outputs."""

    target: type[OracleRriCacheAppender] = Field(
        default_factory=_target_cache_appender,
        exclude=True,
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheConfig = Field(default_factory=OracleRriCacheConfig)
    """Cache path configuration."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration used for metadata validation."""

    dataset: AseEfmDatasetConfig | None = None
    """Optional dataset config snapshot used when initializing metadata."""

    backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """EVL backbone configuration used for metadata validation (optional)."""

    include_backbone: bool = True
    """Whether appended samples include EVL backbone outputs."""

    include_depths: bool = True
    """Whether appended samples include depth maps."""

    include_pointclouds: bool = True
    """Whether appended samples include candidate point clouds."""

    allow_mismatch: bool = False
    """Allow appending samples that don't match cached config signatures."""

    create_if_missing: bool = False
    """Create a new cache metadata/index if missing."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""


class OracleRriCacheDatasetConfig(BaseConfig["OracleRriCacheDataset"]):
    """Configuration for loading cached oracle outputs."""

    target: type[OracleRriCacheDataset] = Field(
        default_factory=_target_cache_dataset,
        exclude=True,
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheConfig = Field(default_factory=OracleRriCacheConfig)
    """Cache path configuration."""

    load_backbone: bool = True
    """Whether to load cached EVL backbone outputs."""

    backbone_keep_fields: list[str] | None = None
    """Optional allowlist of EVL backbone fields to decode from cache."""

    map_location: torch.device = Field(default="cpu")
    """Device string for loading cached tensors (e.g., 'cpu', 'cuda')."""

    include_efm_snippet: bool = True
    """Wether to include the EFM snippet in the loaded samples."""

    include_gt_mesh: bool = False
    """Wether to include the ground truth mesh in the loaded samples. Applies only if ``include_efm_snippet=True``."""

    load_candidates: bool = True
    """Whether to decode candidate sampling metadata."""

    load_depths: bool = True
    """Whether to decode candidate depth maps (needed for VIN training)."""

    load_candidate_pcs: bool = True
    """Whether to decode candidate point clouds."""

    efm_keep_keys: list[str] | None = None
    """Optional allowlist of EFM keys to keep when loading snippets."""

    return_format: Literal["cache_sample", "vin_batch"] = "cache_sample"
    """Return cached samples or VIN-ready batches."""

    simplification: float | None = None
    """Optional fraction of cached samples to expose (for quick validation)."""

    train_val_split: float = 0.2
    """Fraction of cached samples to reserve for validation when splitting."""

    split: Literal["all", "train", "val"] = "all"
    """Subset of cached entries to load (``train``/``val`` use split indices)."""

    limit: int | None = None
    """Optional cap on number of cached samples loaded."""

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str) -> torch.device:
        return cls._resolve_device(value)

    @field_validator("train_val_split", mode="before")
    @classmethod
    def _validate_train_val_split(cls, value: float) -> float:
        split = float(value)
        if not 0.0 <= split <= 1.0:
            raise ValueError("train_val_split must be in [0, 1].")
        return split

    @field_validator("simplification", mode="before")
    @classmethod
    def _validate_simplification(cls, value: float | None) -> float | None:
        if value is None:
            return None
        simplification = float(value)
        if not 0.0 < simplification < 1.0:
            raise ValueError("simplification must be in (0, 1).")
        return simplification


def _read_cache_index(path: Path, *, allow_missing: bool = False) -> list[OracleRriCacheEntry]:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing cache index at {path}")
    entries: list[OracleRriCacheEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        entries.append(
            OracleRriCacheEntry(
                key=item["key"],
                scene_id=item["scene_id"],
                snippet_id=item["snippet_id"],
                path=item["path"],
            ),
        )
    return entries


def _serialize_cache_entries(entries: list[OracleRriCacheEntry]) -> str:
    if not entries:
        return ""
    return "\n".join(json.dumps(asdict(entry)) for entry in entries) + "\n"


def _write_cache_index(path: Path, entries: list[OracleRriCacheEntry]) -> None:
    payload = _serialize_cache_entries(entries)
    path.write_text(payload, encoding="utf-8")


def _write_cache_index_if_changed(path: Path, entries: list[OracleRriCacheEntry]) -> None:
    payload = _serialize_cache_entries(entries)
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.write_text(payload, encoding="utf-8")


def _ensure_train_val_split(
    *,
    index_path: Path,
    train_index_path: Path,
    val_index_path: Path,
    val_fraction: float,
    console: Console | None,
) -> tuple[list[OracleRriCacheEntry], list[OracleRriCacheEntry]]:
    base_entries = _read_cache_index(index_path)
    if not base_entries:
        _write_cache_index_if_changed(train_index_path, [])
        _write_cache_index_if_changed(val_index_path, [])
        return [], []

    val_fraction = max(0.0, min(1.0, float(val_fraction)))
    if val_fraction <= 0.0:
        _write_cache_index_if_changed(train_index_path, base_entries)
        _write_cache_index_if_changed(val_index_path, [])
        return base_entries, []
    if val_fraction >= 1.0:
        _write_cache_index_if_changed(train_index_path, [])
        _write_cache_index_if_changed(val_index_path, base_entries)
        return [], base_entries

    entry_by_key = {entry.key: entry for entry in base_entries}
    train_entries: list[OracleRriCacheEntry] = []
    val_entries: list[OracleRriCacheEntry] = []
    used_keys: set[str] = set()

    if train_index_path.exists() and val_index_path.exists():
        for entry in _read_cache_index(train_index_path, allow_missing=True):
            if entry.key in entry_by_key and entry.key not in used_keys:
                train_entries.append(entry_by_key[entry.key])
                used_keys.add(entry.key)
        for entry in _read_cache_index(val_index_path, allow_missing=True):
            if entry.key in entry_by_key and entry.key not in used_keys:
                val_entries.append(entry_by_key[entry.key])
                used_keys.add(entry.key)

    missing_entries = [entry for entry in base_entries if entry.key not in used_keys]
    target_val = int(round(len(base_entries) * val_fraction))
    if val_entries and len(val_entries) != target_val and console is not None:
        console.warn(
            "Existing train/val split size differs from train_val_split; preserving prior assignments.",
        )

    for entry in missing_entries:
        if len(val_entries) < target_val:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    _write_cache_index_if_changed(train_index_path, train_entries)
    _write_cache_index_if_changed(val_index_path, val_entries)
    return train_entries, val_entries


def _load_efm_snippet_for_cache(
    *,
    scene_id: str,
    snippet_id: str,
    dataset_payload: dict[str, Any] | None,
    device: str,
    paths: PathConfig,
    include_gt_mesh: bool,
) -> EfmSnippetView:
    payload = dict(dataset_payload or {})
    payload["paths"] = payload.get("paths", paths)
    payload["scene_ids"] = [scene_id]
    payload["snippet_ids"] = [snippet_id]
    payload["batch_size"] = 1
    payload["device"] = device
    payload["wds_shuffle"] = False
    payload["wds_repeat"] = False
    payload["load_meshes"] = bool(include_gt_mesh)
    payload.setdefault("require_mesh", False)
    payload["verbosity"] = Verbosity.QUIET
    cfg = AseEfmDatasetConfig(**payload)
    dataset = cfg.setup_target()
    return next(iter(dataset))


class _EfmSnippetLoader:
    """Persistent per-worker loader for on-demand EFM snippets."""

    def __init__(
        self,
        *,
        dataset_payload: dict[str, Any] | None,
        device: str,
        paths: PathConfig,
        include_gt_mesh: bool,
    ) -> None:
        self._dataset_payload = dict(dataset_payload or {})
        self._device = str(device)
        self._paths = paths
        self._include_gt_mesh = include_gt_mesh
        self._datasets: dict[str, AseEfmDataset] = {}

    def load(self, *, scene_id: str, snippet_id: str) -> EfmSnippetView:
        """Load a snippet view, reusing a cached dataset per scene."""
        dataset = self._datasets.get(scene_id)
        if dataset is None:
            payload = dict(self._dataset_payload)
            payload["paths"] = payload.get("paths", self._paths)
            payload["scene_ids"] = [scene_id]
            payload["snippet_ids"] = []
            payload["snippet_key_filter"] = []
            payload["batch_size"] = 1
            payload["device"] = self._device
            payload["wds_shuffle"] = False
            payload["wds_repeat"] = False
            payload["load_meshes"] = bool(self._include_gt_mesh)
            payload.setdefault("require_mesh", False)
            payload["verbosity"] = Verbosity.QUIET
            cfg = AseEfmDatasetConfig(**payload)
            dataset = cfg.setup_target()
            self._datasets[scene_id] = dataset

        dataset._snippet_key_filter = {snippet_id}  # type: ignore[attr-defined]
        for sample in dataset:
            return sample
        raise FileNotFoundError(
            f"Failed to locate snippet={snippet_id} in scene={scene_id}.",
        )


# ----------------------------------------------------------------------------- Cache IO helpers
def build_cache_payload(
    label_batch: OracleRriLabelBatch,
    *,
    backbone_out: EvlBackboneOutput | None,
    include_backbone: bool,
    include_depths: bool,
    include_pointclouds: bool,
) -> dict[str, Any]:
    """Serialize a label batch and optional backbone output.

    Args:
        label_batch: Oracle label batch to serialize.
        backbone_out: Optional EVL backbone outputs.
        include_backbone: Whether to include backbone outputs.
        include_depths: Whether to include candidate depth maps.
        include_pointclouds: Whether to include candidate point clouds.

    Returns:
        Dictionary payload ready for torch.save.
    """
    payload: dict[str, Any] = {
        "scene_id": label_batch.sample.scene_id,
        "snippet_id": label_batch.sample.snippet_id,
        "candidates": encode_candidates(label_batch.candidates),
        "depths": encode_depths(label_batch.depths) if include_depths else None,
        "candidate_pcs": encode_candidate_pcs(label_batch.candidate_pcs) if include_pointclouds else None,
        "rri": encode_rri(label_batch.rri),
    }
    if include_backbone:
        if backbone_out is None:
            raise ValueError(
                "include_backbone=True requires backbone_out to be provided.",
            )
        payload["backbone"] = encode_backbone(backbone_out)
    return payload


# ----------------------------------------------------------------------------- Cache writer
class OracleRriCacheWriter:
    """Build an offline cache of oracle labels (and optional EVL outputs)."""

    def __init__(self, config: OracleRriCacheWriterConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(
            self.config.verbosity,
        )
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
        """Iterate the dataset and persist cached outputs.

        Returns:
            List of index entries for cached samples created in this run.
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
            existing_entries = _read_cache_index(index_path, allow_missing=True)
            valid_entries: list[OracleRriCacheEntry] = []
            for entry in existing_entries:
                entry_path = self.config.cache.cache_dir / entry.path
                if entry_path.exists():
                    valid_entries.append(entry)
                    existing_pairs.add((entry.scene_id, entry.snippet_id))
                else:
                    self.console.warn(
                        f"Index entry missing sample file; will regenerate if encountered "
                        f"(scene={entry.scene_id} snippet={entry.snippet_id}).",
                    )
            existing_entries = valid_entries
            if existing_entries:
                self.console.log(
                    f"Found {len(existing_entries)} existing cached samples; skipping duplicates by scene/snippet.",
                )
            index_path.unlink()

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
        if existing_entries:
            _write_cache_index(index_path, existing_entries)
        else:
            index_path.touch()
        with index_path.open("a", encoding="utf-8") as index_f:
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
                    index_f.write(json.dumps(asdict(cache_entry)) + "\n")
                    count += 1
                    existing_pairs.add(sample_pair)
                    if max_samples is not None and count >= max_samples:
                        break
                except Exception as e:
                    self.console.error(
                        f"Failed to cache scene={sample.scene_id} snippet={sample.snippet_id}: {e}",
                    )
                    fail_count += 1
                    continue
                finally:
                    if fail_count > self.config.num_failures_allowed:
                        raise RuntimeError(
                            f"Exceeded maximum allowed failures ({self.config.num_failures_allowed}). Aborting cache write.",
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
        """Cache a single sample.

        Args:
            sample: Input snippet with meshes and semidense points.
            samples_dir: Directory where sample payloads are written.

        Returns:
            Index entry for the cached sample.
        """
        label_batch = self._labeler.run(sample)
        backbone_out = None
        if self._backbone is not None:
            backbone_out = self._backbone.forward(sample.efm)

        payload = self._encode_sample(label_batch, backbone_out=backbone_out)

        config_hash = self._config_hash or "unknown"
        base_key = _format_sample_key(sample.scene_id, sample.snippet_id, config_hash)
        sample_key, sample_path = _unique_sample_path(samples_dir, base_key)
        torch.save(payload, sample_path)

        self.console.log(
            f"Cached scene={sample.scene_id} snippet={sample.snippet_id} -> {sample_path.name}",
        )
        return OracleRriCacheEntry(
            key=sample_key,
            scene_id=sample.scene_id,
            snippet_id=sample.snippet_id,
            path=str(sample_path.relative_to(self.config.cache.cache_dir)),
        )

    def _encode_sample(
        self,
        label_batch: OracleRriLabelBatch,
        *,
        backbone_out: EvlBackboneOutput | None,
    ) -> dict[str, Any]:
        """Serialize a label batch and optional backbone output."""
        return build_cache_payload(
            label_batch,
            backbone_out=backbone_out,
            include_backbone=self.config.include_backbone,
            include_depths=self.config.include_depths,
            include_pointclouds=self.config.include_pointclouds,
        )


# ----------------------------------------------------------------------------- Cache appender
class OracleRriCacheAppender:
    """Append cached oracle outputs to an existing cache directory."""

    def __init__(self, config: OracleRriCacheAppenderConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(
            self.config.verbosity,
        )

        if self.config.include_backbone and self.config.backbone is None:
            raise ValueError("include_backbone=True requires a backbone config.")

        self._meta = self._prepare_metadata()
        self._config_hash = self._meta.config_hash or "unknown"
        self._index_path = self.config.cache.index_path
        self._samples_dir = self.config.cache.samples_dir
        self._num_samples = self._resolve_num_samples()

    def append(
        self,
        label_batch: OracleRriLabelBatch,
        *,
        backbone_out: EvlBackboneOutput | None,
    ) -> OracleRriCacheEntry:
        """Append a single oracle label batch to the cache.

        Args:
            label_batch: Oracle label batch to cache.
            backbone_out: Optional EVL backbone outputs (required if include_backbone=True).

        Returns:
            The cache index entry for the appended sample.
        """
        payload = build_cache_payload(
            label_batch,
            backbone_out=backbone_out,
            include_backbone=self.config.include_backbone,
            include_depths=self.config.include_depths,
            include_pointclouds=self.config.include_pointclouds,
        )

        config_hash = self._config_hash or "unknown"
        base_key = _format_sample_key(
            label_batch.sample.scene_id,
            label_batch.sample.snippet_id,
            config_hash,
        )
        sample_key, sample_path = _unique_sample_path(self._samples_dir, base_key)
        torch.save(payload, sample_path)

        entry = OracleRriCacheEntry(
            key=sample_key,
            scene_id=label_batch.sample.scene_id,
            snippet_id=label_batch.sample.snippet_id,
            path=str(sample_path.relative_to(self.config.cache.cache_dir)),
        )
        with self._index_path.open("a", encoding="utf-8") as index_f:
            index_f.write(json.dumps(asdict(entry)) + "\n")

        self._num_samples += 1
        self._meta.num_samples = self._num_samples
        _write_metadata(self.config.cache.metadata_path, self._meta)
        self.console.log(
            f"Appended scene={entry.scene_id} snippet={entry.snippet_id} -> {sample_path.name} (count={self._num_samples})",
        )
        return entry

    def _prepare_metadata(self) -> OracleRriCacheMetadata:
        cache_dir = self.config.cache.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache.samples_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.config.cache.index_path
        if not index_path.exists():
            index_path.write_text("", encoding="utf-8")

        meta_path = self.config.cache.metadata_path
        if meta_path.exists():
            meta = _read_metadata(meta_path)
            self._validate_metadata(meta)
            meta.config_hash = _ensure_config_hash(
                meta,
                include_backbone=self.config.include_backbone,
                include_depths=self.config.include_depths,
                include_pointclouds=self.config.include_pointclouds,
            )
            _write_metadata(meta_path, meta)
            return meta

        if not self.config.create_if_missing:
            raise FileNotFoundError(
                f"Missing cache metadata at {meta_path} (set create_if_missing=True to initialize).",
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
        meta.num_samples = 0
        _write_metadata(meta_path, meta)
        return meta

    def _resolve_num_samples(self) -> int:
        if self._meta.num_samples is not None:
            return int(self._meta.num_samples)
        index_path = self.config.cache.index_path
        if not index_path.exists():
            return 0
        return sum(1 for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _validate_metadata(self, meta: OracleRriCacheMetadata) -> None:
        labeler_snapshot = snapshot_config(self.config.labeler)
        labeler_sig = _config_signature(labeler_snapshot)
        if meta.labeler_signature != labeler_sig:
            msg = "Labeler config signature mismatch for cache append."
            if self.config.allow_mismatch:
                self.console.warn(msg)
            else:
                raise ValueError(msg)

        if self.config.include_backbone and self.config.backbone is None:
            raise ValueError("include_backbone=True requires a backbone config.")

        if meta.include_backbone is not None and meta.include_backbone != self.config.include_backbone:
            msg = "include_backbone mismatch between cache metadata and append config."
            if self.config.allow_mismatch:
                self.console.warn(msg)
            else:
                raise ValueError(msg)

        if meta.include_depths is not None and meta.include_depths != self.config.include_depths:
            msg = "include_depths mismatch between cache metadata and append config."
            if self.config.allow_mismatch:
                self.console.warn(msg)
            else:
                raise ValueError(msg)

        if meta.include_pointclouds is not None and meta.include_pointclouds != self.config.include_pointclouds:
            msg = "include_pointclouds mismatch between cache metadata and append config."
            if self.config.allow_mismatch:
                self.console.warn(msg)
            else:
                raise ValueError(msg)

        if self.config.include_backbone:
            backbone_snapshot = snapshot_config(self.config.backbone) if self.config.backbone else None
            backbone_sig = _config_signature(backbone_snapshot) if backbone_snapshot else None
            if meta.backbone_signature and meta.backbone_signature != backbone_sig:
                msg = "Backbone config signature mismatch for cache append."
                if self.config.allow_mismatch:
                    self.console.warn(msg)
                else:
                    raise ValueError(msg)


# ----------------------------------------------------------------------------- Cache dataset
class OracleRriCacheDataset(Dataset[OracleRriCacheSample]):
    """Map-style dataset that reads cached oracle outputs."""

    def __init__(self, config: OracleRriCacheDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        if self.config.include_gt_mesh and not self.config.include_efm_snippet:
            self.console.warn(
                "include_gt_mesh=True has no effect unless include_efm_snippet=True.",
            )
        self._index = self._load_index()
        self.metadata = _read_metadata(self.config.cache.metadata_path)
        self._warned_worker_cuda = False
        self._len = self._resolve_len()
        self._efm_loader_by_device: dict[str, _EfmSnippetLoader] = {}

    def _resolve_len(self) -> int:
        base_len = len(self._index)
        simplification = self.config.simplification
        if simplification is not None:
            base_len = int(base_len * simplification)
        limit = self.config.limit
        if limit is not None:
            return min(base_len, int(limit))
        return base_len

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["console"] = None
        state["_efm_loader_by_device"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.__dict__.get("console") is None:
            self.console = Console.with_prefix(self.__class__.__name__)
        if self.__dict__.get("_warned_worker_cuda") is None:
            self._warned_worker_cuda = False
        if self.__dict__.get("_efm_loader_by_device") is None:
            self._efm_loader_by_device = {}

    def _resolve_map_location(self) -> str:
        map_location = str(self.config.map_location)
        worker_info = get_worker_info()
        if worker_info is None:
            return map_location
        try:
            device = torch.device(map_location)
        except (TypeError, ValueError):
            return "cpu"
        if device.type == "cuda":
            if not self._warned_worker_cuda:
                self.console.warn(
                    "Cache DataLoader worker cannot load CUDA tensors; forcing map_location='cpu'.",
                )
                self._warned_worker_cuda = True
            return "cpu"
        return map_location

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> OracleRriCacheSample | "VinOracleBatch":  # type: ignore
        """Load and decode a cached sample by index.

        Args:
            idx: Index into the cache.

        Returns:
            Decoded cache sample with oracle outputs.
        """
        if idx < 0 or idx >= self._len:
            raise IndexError("Cache index out of range.")
        entry = self._index[idx]
        path = self.config.cache.cache_dir / entry.path
        map_location = self._resolve_map_location()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            payload = torch.load(path, map_location=map_location, weights_only=False)
        device = torch.device(map_location)

        require_cache_sample = self.config.return_format == "cache_sample"
        require_depths = self.config.return_format in {"cache_sample", "vin_batch"}
        require_candidates = require_cache_sample
        require_pcs = require_cache_sample

        if require_candidates and not self.config.load_candidates:
            raise ValueError("load_candidates must be True when return_format='cache_sample'.")
        if require_depths and not self.config.load_depths:
            raise ValueError("load_depths must be True when return_format requires depths.")
        if require_pcs and not self.config.load_candidate_pcs:
            raise ValueError("load_candidate_pcs must be True when return_format='cache_sample'.")

        candidates = None
        if self.config.load_candidates:
            candidates = decode_candidates(payload["candidates"])
        depths = None
        if self.config.load_depths and payload.get("depths") is not None:
            depths = decode_depths(payload["depths"], device=device)
        pcs = None
        if self.config.load_candidate_pcs and payload.get("candidate_pcs") is not None:
            pcs = decode_candidate_pcs(payload["candidate_pcs"], device=device)
        rri = decode_rri(payload["rri"], device=device)

        if not self.config.load_candidates:
            payload.pop("candidates", None)
        if not self.config.load_candidate_pcs:
            payload.pop("candidate_pcs", None)
        if not self.config.load_depths:
            payload.pop("depths", None)

        if require_depths and depths is None:
            msg = "Cached sample missing depths; re-generate with include_depths=True."
            raise ValueError(msg)
        if require_pcs and pcs is None:
            msg = "Cached sample missing candidate point clouds; re-generate with include_pointclouds=True."
            raise ValueError(msg)

        backbone_out = None
        if self.config.load_backbone and payload.get("backbone") is not None:
            keep_fields = None
            if self.config.backbone_keep_fields:
                keep_fields = set(self.config.backbone_keep_fields)
            backbone_out = decode_backbone(
                payload["backbone"],
                device=device,
                include_fields=keep_fields,
            )

        efm_snippet = None
        if self.config.include_efm_snippet:
            try:
                loader = self._efm_loader_by_device.get(map_location)
                if loader is None:
                    loader = _EfmSnippetLoader(
                        dataset_payload=self.metadata.dataset_config,
                        device=map_location,
                        paths=self.config.paths,
                        include_gt_mesh=self.config.include_gt_mesh,
                    )
                    self._efm_loader_by_device[map_location] = loader
                efm_snippet = loader.load(
                    scene_id=payload["scene_id"],
                    snippet_id=payload["snippet_id"],
                )
                if self.config.include_gt_mesh and efm_snippet.mesh is None:
                    self.console.warn(
                        f"GT mesh missing for scene={payload['scene_id']} snippet={payload['snippet_id']}.",
                    )
                if self.config.efm_keep_keys is not None:
                    efm_snippet = efm_snippet.prune_efm(set(self.config.efm_keep_keys))
            except Exception as exc:
                self.console.warn(
                    f"Failed to load EFM snippet for scene={payload['scene_id']} "
                    f"snippet={payload['snippet_id']}: {exc}",
                )
                efm_snippet = None

        if self.config.return_format == "vin_batch":
            if depths is None:
                raise ValueError(
                    "VIN batch requires depths to be loaded. Set load_depths=True.",
                )
            return self._to_vin_batch_from_parts(
                efm_snippet=efm_snippet,
                depths=depths,
                rri=rri,
                scene_id=payload["scene_id"],
                snippet_id=payload["snippet_id"],
                backbone_out=backbone_out,
            )

        cache_sample = OracleRriCacheSample(
            key=entry.key,
            scene_id=payload["scene_id"],
            snippet_id=payload["snippet_id"],
            candidates=candidates,
            depths=depths,
            candidate_pcs=pcs,
            rri=rri,
            backbone_out=backbone_out,
            efm_snippet_view=efm_snippet,
        )
        return cache_sample

    def _to_vin_batch(self, cache_sample: OracleRriCacheSample) -> "VinOracleBatch":
        from ..lightning.lit_datamodule import VinOracleBatch

        rri = cache_sample.rri
        depths = cache_sample.depths
        return VinOracleBatch(
            efm_snippet_view=cache_sample.efm_snippet_view,
            candidate_poses_world_cam=depths.poses,
            reference_pose_world_rig=depths.reference_pose,
            rri=rri.rri,
            pm_dist_before=rri.pm_dist_before,
            pm_dist_after=rri.pm_dist_after,
            pm_acc_before=rri.pm_acc_before,
            pm_comp_before=rri.pm_comp_before,
            pm_acc_after=rri.pm_acc_after,
            pm_comp_after=rri.pm_comp_after,
            p3d_cameras=depths.p3d_cameras,
            scene_id=cache_sample.scene_id,
            snippet_id=cache_sample.snippet_id,
            backbone_out=cache_sample.backbone_out,
        )

    def _to_vin_batch_from_parts(
        self,
        *,
        efm_snippet: EfmSnippetView | None,
        depths: CandidateDepths,
        rri: RriResult,
        scene_id: str,
        snippet_id: str,
        backbone_out: EvlBackboneOutput | None,
    ) -> "VinOracleBatch":
        from ..lightning.lit_datamodule import VinOracleBatch

        return VinOracleBatch(
            efm_snippet_view=efm_snippet,
            candidate_poses_world_cam=depths.poses,
            reference_pose_world_rig=depths.reference_pose,
            rri=rri.rri,
            pm_dist_before=rri.pm_dist_before,
            pm_dist_after=rri.pm_dist_after,
            pm_acc_before=rri.pm_acc_before,
            pm_comp_before=rri.pm_comp_before,
            pm_acc_after=rri.pm_acc_after,
            pm_comp_after=rri.pm_comp_after,
            p3d_cameras=depths.p3d_cameras,
            scene_id=scene_id,
            snippet_id=snippet_id,
            backbone_out=backbone_out,
        )

    def append_entry(self, entry: OracleRriCacheEntry) -> None:
        """Update the in-memory index with a newly appended entry."""
        self._index.append(entry)
        if self.metadata.num_samples is not None:
            self.metadata.num_samples = len(self._index)

    def _load_index(self) -> list[OracleRriCacheEntry]:
        """Load cache index entries from JSONL."""
        if self.config.split == "all":
            return _read_cache_index(self.config.cache.index_path)

        train_entries, val_entries = _ensure_train_val_split(
            index_path=self.config.cache.index_path,
            train_index_path=self.config.cache.train_index_path,
            val_index_path=self.config.cache.val_index_path,
            val_fraction=self.config.train_val_split,
            console=self.console,
        )
        return train_entries if self.config.split == "train" else val_entries


def rebuild_cache_index(
    *,
    cache_dir: Path,
    train_val_split: float | None = None,
    rng_seed: int | None = None,
) -> int:
    """Rebuild index.jsonl and train/val splits from cached sample filenames."""
    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig())
    samples_dir = cache_cfg.samples_dir
    if not samples_dir.exists():
        return 0
    sample_paths = sorted(samples_dir.glob("*.pt"))
    entries: list[OracleRriCacheEntry] = []
    for sample_path in sample_paths:
        stem = sample_path.stem
        base = stem.split("__", 1)[0]
        tokens = base.split("_")
        scene_id = ""
        snippet_id = ""
        if len(tokens) >= 6 and tokens[0:3] == ["ASE", "NBV", "SNIPPET"]:
            scene_id = tokens[3]
            snippet_id = "_".join(tokens[4:-1])
        if not snippet_id:
            snippet_id = "unknown"
        entries.append(
            OracleRriCacheEntry(
                key=stem,
                scene_id=scene_id,
                snippet_id=snippet_id,
                path=str(sample_path.relative_to(cache_dir)),
            ),
        )
    _write_cache_index(cache_cfg.index_path, entries)

    val_fraction = (
        float(train_val_split) if train_val_split is not None else float(OracleRriCacheDatasetConfig().train_val_split)
    )
    val_fraction = max(0.0, min(1.0, val_fraction))
    rng = random.Random(rng_seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_fraction))
    val_entries = shuffled[:val_count]
    train_entries = shuffled[val_count:]
    _write_cache_index(cache_cfg.train_index_path, train_entries)
    _write_cache_index(cache_cfg.val_index_path, val_entries)
    meta_path = cache_cfg.metadata_path
    if meta_path.exists():
        meta = _read_metadata(meta_path)
        meta.num_samples = len(entries)
        _write_metadata(meta_path, meta)
    return len(entries)


__all__ = [
    "CACHE_VERSION",
    "OracleRriCacheAppender",
    "OracleRriCacheAppenderConfig",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "decode_backbone",
    "decode_candidate_pcs",
    "decode_candidates",
    "decode_depths",
    "decode_rri",
    "encode_backbone",
    "encode_candidate_pcs",
    "encode_candidates",
    "encode_depths",
    "encode_rri",
    "snapshot_config",
    "snapshot_dataset_config",
    "rebuild_cache_index",
]

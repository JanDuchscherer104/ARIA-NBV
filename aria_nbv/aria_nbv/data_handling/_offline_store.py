"""Low-level storage primitives for the VIN offline dataset format.

This module owns the immutable on-disk layout of the VIN offline dataset:

- path and split configuration,
- per-shard block materialization helpers,
- manifest and sample-index loading, and
- mmap-based random-access reads for fixed-size tensor blocks.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, ValidationInfo, field_validator

from ..configs import PathConfig
from ..utils import BaseConfig
from ._offline_format import (
    VinOfflineBlockSpec,
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineShardSpec,
    read_sample_index,
)

OFFLINE_DATASET_VERSION = 1
"""Version of the immutable VIN offline dataset format."""


def _safe_block_name(name: str) -> str:
    """Convert a logical block name into a filesystem-safe stem.

    Args:
        name: Logical block name, for example ``"vin.points_world"``.

    Returns:
        Filesystem-safe block stem.
    """

    return name.replace("/", "__").replace(".", "__")


class VinOfflineStoreConfig(BaseConfig):
    """Filesystem configuration for one immutable VIN offline dataset."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    store_dir: Path = Field(default_factory=lambda: PathConfig().offline_cache_dir / "vin_offline")
    """Root directory containing the immutable VIN offline dataset."""

    manifest_filename: str = "manifest.json"
    """Filename of the top-level manifest."""

    sample_index_filename: str = "sample_index.jsonl"
    """Filename of the global sample index."""

    shards_dirname: str = "shards"
    """Directory containing immutable shard subdirectories."""

    splits_dirname: str = "splits"
    """Directory containing split membership arrays."""

    @field_validator("store_dir", mode="before")
    @classmethod
    def _resolve_store_dir(cls, value: str | Path, info: ValidationInfo) -> Path:
        """Resolve relative dataset directories against project data roots.

        Args:
            value: Raw path value.
            info: Pydantic validation context.

        Returns:
            Resolved absolute dataset directory.
        """

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
    def manifest_path(self) -> Path:
        """Return the absolute manifest path."""

        return self.store_dir / self.manifest_filename

    @property
    def sample_index_path(self) -> Path:
        """Return the absolute sample-index path."""

        return self.store_dir / self.sample_index_filename

    @property
    def shards_dir(self) -> Path:
        """Return the absolute shard root directory."""

        return self.store_dir / self.shards_dirname

    @property
    def splits_dir(self) -> Path:
        """Return the absolute split-array directory."""

        return self.store_dir / self.splits_dirname

    def split_path(self, split: str) -> Path:
        """Return the split-array path for one split.

        Args:
            split: Split name such as ``"all"``, ``"train"``, or ``"val"``.

        Returns:
            Absolute split-array path.
        """

        return self.splits_dir / f"{split}.npy"


def write_split_indices(config: VinOfflineStoreConfig, split_to_indices: dict[str, np.ndarray]) -> None:
    """Persist split membership arrays.

    Args:
        config: Store configuration.
        split_to_indices: Split membership arrays keyed by split name.
    """

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    for split, indices in split_to_indices.items():
        np.save(config.split_path(split), np.asarray(indices, dtype=np.int64), allow_pickle=False)


def read_split_indices(config: VinOfflineStoreConfig, split: str) -> np.ndarray:
    """Load the global sample indices for one split.

    Args:
        config: Store configuration.
        split: Split name such as ``"all"``, ``"train"``, or ``"val"``.

    Returns:
        Global sample indices for the requested split.
    """

    return np.load(config.split_path(split), allow_pickle=False)


def write_fixed_block(shard_dir: Path, name: str, array: np.ndarray) -> VinOfflineBlockSpec:
    """Write one fixed-size numeric block for a shard.

    Args:
        shard_dir: Destination shard directory.
        name: Logical block name.
        array: NumPy array to store.

    Returns:
        Block descriptor for the stored array.
    """

    stem = _safe_block_name(name)
    rel_path = f"{stem}.npy"
    np.save(shard_dir / rel_path, array, allow_pickle=False)
    return VinOfflineBlockSpec(
        name=name,
        kind="fixed_npy",
        paths=[rel_path],
        dtype=str(array.dtype),
        shape=list(array.shape),
        optional=False,
    )


def write_pickle_records(shard_dir: Path, name: str, records: list[Any]) -> VinOfflineBlockSpec:
    """Write one per-row diagnostic record list for a shard.

    Args:
        shard_dir: Destination shard directory.
        name: Logical block name.
        records: Per-row Python objects to pickle.

    Returns:
        Block descriptor for the stored record list.
    """

    stem = _safe_block_name(name)
    rel_path = f"{stem}.pkl"
    with (shard_dir / rel_path).open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return VinOfflineBlockSpec(
        name=name,
        kind="pickle_records",
        paths=[rel_path],
        dtype=None,
        shape=[len(records)],
        optional=True,
    )


@dataclass(slots=True)
class OpenedShard:
    """Worker-local opened shard state."""

    spec: VinOfflineShardSpec
    """Shard descriptor backing the opened state."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    """Memory-mapped numeric blocks keyed by logical block name."""

    record_lists: dict[str, list[Any]] = field(default_factory=dict)
    """Lazy-loaded diagnostic record lists keyed by logical block name."""


class VinOfflineStoreReader:
    """Read immutable VIN offline datasets with mmap-backed random access."""

    def __init__(self, config: VinOfflineStoreConfig) -> None:
        """Load the manifest, sample index, and split metadata.

        Args:
            config: Store configuration pointing at an immutable dataset.
        """

        self.config = config
        self.manifest = VinOfflineManifest.read(config.manifest_path)
        self.sample_index = read_sample_index(config.sample_index_path)
        self._records_by_sample_index = {record.sample_index: record for record in self.sample_index}
        self._shards = {spec.shard_id: spec for spec in self.manifest.shards}
        self._opened: dict[str, OpenedShard] = {}
        self._split_cache: dict[str, np.ndarray] = {}

    def get_split_records(self, split: str) -> list[VinOfflineIndexRecord]:
        """Return index records for the requested split.

        Args:
            split: Split name such as ``"all"``, ``"train"``, or ``"val"``.

        Returns:
            Ordered index records for the split.
        """

        if split not in self._split_cache:
            self._split_cache[split] = read_split_indices(self.config, split)
        return [self._records_by_sample_index[int(idx)] for idx in self._split_cache[split]]

    def _open_shard(self, shard_id: str) -> OpenedShard:
        """Open one shard and cache its mmap-backed blocks.

        Args:
            shard_id: Stable shard identifier.

        Returns:
            Worker-local opened shard handle.
        """

        opened = self._opened.get(shard_id)
        if opened is not None:
            return opened

        spec = self._shards[shard_id]
        shard_dir = self.config.store_dir / spec.relative_dir
        opened = OpenedShard(spec=spec)
        for block_name, block_spec in spec.blocks.items():
            if block_spec.kind == "fixed_npy":
                opened.arrays[block_name] = np.load(shard_dir / block_spec.paths[0], mmap_mode="r", allow_pickle=False)
        self._opened[shard_id] = opened
        return opened

    def _load_record_list(self, opened: OpenedShard, block_name: str) -> list[Any]:
        """Load a per-row diagnostic record list for one shard.

        Args:
            opened: Worker-local opened shard handle.
            block_name: Logical block name to load.

        Returns:
            Per-row record list for the block.
        """

        if block_name in opened.record_lists:
            return opened.record_lists[block_name]
        block = opened.spec.blocks[block_name]
        shard_dir = self.config.store_dir / opened.spec.relative_dir
        with (shard_dir / block.paths[0]).open("rb") as handle:
            opened.record_lists[block_name] = pickle.load(handle)
        return opened.record_lists[block_name]

    def read_numeric_block(self, record: VinOfflineIndexRecord, block_name: str) -> np.ndarray:
        """Read one numeric block row for a sample.

        Args:
            record: Global sample-index record.
            block_name: Logical block name.

        Returns:
            NumPy array view for the requested sample row.
        """

        opened = self._open_shard(record.shard_id)
        return np.asarray(opened.arrays[block_name][record.row])

    def read_optional_record(self, record: VinOfflineIndexRecord, block_name: str) -> Any | None:
        """Read one optional per-row diagnostic record.

        Args:
            record: Global sample-index record.
            block_name: Logical block name.

        Returns:
            Stored per-row Python object or ``None``.
        """

        opened = self._open_shard(record.shard_id)
        if block_name not in opened.spec.blocks:
            return None
        records = self._load_record_list(opened, block_name)
        return records[record.row]


__all__ = [
    "OFFLINE_DATASET_VERSION",
    "OpenedShard",
    "VinOfflineStoreConfig",
    "VinOfflineStoreReader",
    "read_split_indices",
    "write_fixed_block",
    "write_pickle_records",
    "write_split_indices",
]

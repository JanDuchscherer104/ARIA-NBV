"""Split and block I/O helpers for the immutable VIN offline dataset."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._offline_format import VinOfflineBlockSpec


class _SplitStoreConfig(Protocol):
    """Minimal store-config contract for split-array persistence."""

    @property
    def splits_dir(self) -> Path: ...

    def split_path(self, split: str) -> Path: ...


def _safe_block_name(name: str) -> str:
    """Convert a logical block name into a filesystem-safe stem."""

    return name.replace("/", "__").replace(".", "__")


def write_split_indices(config: _SplitStoreConfig, split_to_indices: dict[str, np.ndarray]) -> None:
    """Persist split membership arrays."""

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    for split, indices in split_to_indices.items():
        np.save(config.split_path(split), np.asarray(indices, dtype=np.int64), allow_pickle=False)


def read_split_indices(config: _SplitStoreConfig, split: str) -> np.ndarray:
    """Load the global sample indices for one split."""

    return np.load(config.split_path(split), allow_pickle=False)


def write_fixed_block(shard_dir: Path, name: str, array: np.ndarray) -> VinOfflineBlockSpec:
    """Write one fixed-size numeric block for a shard."""

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
    """Write one per-row diagnostic record list for a shard."""

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


__all__ = [
    "read_split_indices",
    "write_fixed_block",
    "write_pickle_records",
    "write_split_indices",
]

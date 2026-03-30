"""Coverage helpers for comparing offline caches against available EFM shards.

This module provides the lightweight scan/report utilities used by app
diagnostics to compare cached ``(scene_id, snippet_id)`` pairs against the raw
ASE/EFM shards on disk without materializing full dataset samples.

Contents:
- tar-path expansion and shard sample-key scanning,
- cache index readers for oracle caches,
- per-scene coverage dataclasses, and
- aggregate coverage computation.
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._efm_selection import expand_tar_paths, matches_snippet_token
from .cache_contracts import OracleRriCacheEntry
from .cache_index import read_index

_SCENE_KEY_RE = re.compile(r"^AriaSyntheticEnvironment_(\d+)_")


def expand_tar_urls(tar_urls: Iterable[str]) -> list[Path]:
    """Expand tar URL entries into concrete shard paths.

    Args:
        tar_urls: Iterable of shard paths or glob patterns.

    Returns:
        Unique resolved shard paths in stable order.
    """

    return expand_tar_paths(tar_urls)


def read_cache_index_entries(index_path: Path) -> list[OracleRriCacheEntry]:
    """Read oracle-cache entries from a JSONL index file.

    Args:
        index_path: Path to an oracle-cache index file.

    Returns:
        Parsed cache entries. Missing files return an empty list.
    """

    return read_index(index_path, entry_type=OracleRriCacheEntry, allow_missing=True)


def snippets_by_scene(entries: Iterable[OracleRriCacheEntry]) -> dict[str, set[str]]:
    """Group cache entries by scene id.

    Args:
        entries: Cache index entries.

    Returns:
        Mapping ``scene_id -> {snippet_id, ...}``.
    """

    out: dict[str, set[str]] = {}
    for entry in entries:
        out.setdefault(entry.scene_id, set()).add(entry.snippet_id)
    return out


def scan_tar_sample_keys(tar_path: Path) -> set[str]:
    """List unique WebDataset sample keys contained in a tar shard.

    Args:
        tar_path: Path to a ``.tar`` shard.

    Returns:
        Set of sample keys contained in the shard.
    """

    keys: set[str] = set()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            name = getattr(member, "name", "")
            if not name:
                continue
            prefix = name.split(".", 1)[0]
            if prefix:
                keys.add(prefix)
    return keys


def scan_dataset_snippets(
    tar_paths: Iterable[Path],
    *,
    snippet_key_filter: Iterable[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, set[str]]:
    """Scan tar shards and return available snippet ids per scene.

    Args:
        tar_paths: Iterable of tar shard paths.
        snippet_key_filter: Optional filter tokens applied to sample keys.
        progress_cb: Optional callback invoked as ``progress_cb(done, total)``.

    Returns:
        Mapping ``scene_id -> {snippet_id, ...}``.
    """

    tokens = {str(token) for token in (snippet_key_filter or []) if str(token)}

    def matches_filter(sample_key: str) -> bool:
        if not tokens:
            return True
        return any(matches_snippet_token(sample_key, token) for token in tokens)

    out: dict[str, set[str]] = {}
    tar_list = list(tar_paths)
    total = len(tar_list)
    for idx, tar_path in enumerate(tar_list):
        scene_from_path = tar_path.parent.name if tar_path.parent.name.isdigit() else None
        keys = scan_tar_sample_keys(tar_path)
        filtered = [key for key in keys if matches_filter(key)]
        if scene_from_path is not None:
            out.setdefault(scene_from_path, set()).update(filtered)
        else:
            for key in filtered:
                match = _SCENE_KEY_RE.match(key)
                scene_id = match.group(1) if match else "unknown"
                out.setdefault(scene_id, set()).add(key)
        if progress_cb is not None:
            progress_cb(idx + 1, total)
    return out


@dataclass(slots=True)
class SceneCoverage:
    """Coverage summary for one scene."""

    scene_id: str
    """Scene identifier."""

    dataset_snippets: int
    """Number of snippets discovered in the raw dataset."""

    cache_train_snippets: int
    """Number of train snippets found in the cache."""

    cache_val_snippets: int
    """Number of validation snippets found in the cache."""

    cache_all_snippets: int
    """Number of unique snippets found across all cache splits."""

    coverage_train: float | None
    """Fraction of dataset snippets covered by the train split."""

    coverage_val: float | None
    """Fraction of dataset snippets covered by the validation split."""

    coverage_all: float | None
    """Fraction of dataset snippets covered by any cache split."""


@dataclass(slots=True)
class CacheCoverageReport:
    """Aggregated coverage report between dataset shards and cache indices."""

    dataset_scenes: int
    """Number of scenes present in the raw dataset scan."""

    dataset_snippets: int
    """Total number of dataset snippets discovered across scenes."""

    cache_train_scenes: int
    """Number of scenes represented in the train split."""

    cache_train_snippets: int
    """Number of train snippets across scenes."""

    cache_val_scenes: int
    """Number of scenes represented in the validation split."""

    cache_val_snippets: int
    """Number of validation snippets across scenes."""

    cache_all_scenes: int
    """Number of scenes represented across all cache splits."""

    cache_all_snippets: int
    """Total number of unique cached snippets."""

    cache_outside_dataset: int
    """Number of cached snippets not present in the raw dataset scan."""

    per_scene: list[SceneCoverage]
    """Per-scene coverage breakdown."""

    def as_rows(self) -> list[dict[str, Any]]:
        """Render per-scene coverage rows for DataFrame display."""

        return [
            {
                "scene_id": row.scene_id,
                "dataset_snippets": row.dataset_snippets,
                "cache_train_snippets": row.cache_train_snippets,
                "cache_val_snippets": row.cache_val_snippets,
                "cache_all_snippets": row.cache_all_snippets,
                "coverage_train": row.coverage_train,
                "coverage_val": row.coverage_val,
                "coverage_all": row.coverage_all,
            }
            for row in self.per_scene
        ]


def compute_cache_coverage(
    *,
    dataset_snippets: Mapping[str, set[str]],
    cache_train_snippets: Mapping[str, set[str]] | None,
    cache_val_snippets: Mapping[str, set[str]] | None,
) -> CacheCoverageReport:
    """Compute coverage statistics between dataset snippets and cached indices.

    Args:
        dataset_snippets: Mapping from scene id to raw snippet ids.
        cache_train_snippets: Mapping for the train cache split.
        cache_val_snippets: Mapping for the validation cache split.

    Returns:
        Aggregate coverage report with per-scene rows.
    """

    train = cache_train_snippets or {}
    val = cache_val_snippets or {}
    all_cache: dict[str, set[str]] = {}
    for scene_id, snippets in train.items():
        all_cache.setdefault(scene_id, set()).update(snippets)
    for scene_id, snippets in val.items():
        all_cache.setdefault(scene_id, set()).update(snippets)

    dataset_scene_ids = set(dataset_snippets.keys())
    cache_outside_dataset = 0
    for scene_id, snippets in all_cache.items():
        known = dataset_snippets.get(scene_id)
        if known is None:
            cache_outside_dataset += len(snippets)
            continue
        cache_outside_dataset += sum(1 for snippet_id in snippets if snippet_id not in known)

    scene_ids = sorted(dataset_scene_ids | set(all_cache.keys()) | set(train.keys()) | set(val.keys()))
    per_scene: list[SceneCoverage] = []
    for scene_id in scene_ids:
        ds_snippets = dataset_snippets.get(scene_id, set())
        train_snippets = train.get(scene_id, set())
        val_snippets = val.get(scene_id, set())
        all_snippets = all_cache.get(scene_id, set())

        ds_count = len(ds_snippets)
        train_count = len(train_snippets)
        val_count = len(val_snippets)
        all_count = len(all_snippets)

        cov_train = (train_count / ds_count) if ds_count else None
        cov_val = (val_count / ds_count) if ds_count else None
        cov_all = (all_count / ds_count) if ds_count else None

        per_scene.append(
            SceneCoverage(
                scene_id=str(scene_id),
                dataset_snippets=ds_count,
                cache_train_snippets=train_count,
                cache_val_snippets=val_count,
                cache_all_snippets=all_count,
                coverage_train=cov_train,
                coverage_val=cov_val,
                coverage_all=cov_all,
            ),
        )

    def nonzero_scenes(mapping: Mapping[str, set[str]]) -> int:
        return sum(1 for snippets in mapping.values() if snippets)

    return CacheCoverageReport(
        dataset_scenes=nonzero_scenes(dataset_snippets),
        dataset_snippets=sum(len(snippets) for snippets in dataset_snippets.values()),
        cache_train_scenes=nonzero_scenes(train),
        cache_train_snippets=sum(len(snippets) for snippets in train.values()),
        cache_val_scenes=nonzero_scenes(val),
        cache_val_snippets=sum(len(snippets) for snippets in val.values()),
        cache_all_scenes=nonzero_scenes(all_cache),
        cache_all_snippets=sum(len(snippets) for snippets in all_cache.values()),
        cache_outside_dataset=cache_outside_dataset,
        per_scene=per_scene,
    )


__all__ = [
    "CacheCoverageReport",
    "SceneCoverage",
    "compute_cache_coverage",
    "expand_tar_urls",
    "read_cache_index_entries",
    "scan_dataset_snippets",
    "scan_tar_sample_keys",
    "snippets_by_scene",
]

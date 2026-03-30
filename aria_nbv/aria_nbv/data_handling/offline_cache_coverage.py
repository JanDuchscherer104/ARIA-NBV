"""Utilities to compare offline cache coverage against available ASE EFM shards.

This module avoids loading full EFM samples. Instead it:
- reads cache indices (JSONL) to discover cached ``(scene_id, snippet_id)`` pairs, and
- scans the ATEK WebDataset shard tar headers to list available sample keys.

The primary consumer is the Streamlit ``offline_stats`` panel.
"""

from __future__ import annotations

import glob
import json
import re
import tarfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache_contracts import OracleRriCacheEntry

_SCENE_KEY_RE = re.compile(r"^AriaSyntheticEnvironment_(\d+)_")


def expand_tar_urls(tar_urls: Iterable[str]) -> list[Path]:
    """Expand tar URL entries into concrete paths.

    Args:
        tar_urls: Iterable of shard paths or glob patterns (absolute or relative).

    Returns:
        List of unique resolved Paths in a stable order.
    """
    expanded: list[Path] = []
    for url in tar_urls:
        url = str(url).strip()
        if not url:
            continue
        if any(ch in url for ch in "*?[]"):
            expanded.extend(Path(match) for match in sorted(glob.glob(url)))
        else:
            expanded.append(Path(url))

    out: list[Path] = []
    seen: set[str] = set()
    for path in expanded:
        try:
            resolved = path.expanduser().resolve()
        except FileNotFoundError:
            resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def read_cache_index_entries(index_path: Path) -> list[OracleRriCacheEntry]:
    """Read cache entries from a JSONL index.

    Args:
        index_path: Path to ``index.jsonl`` / ``train_index.jsonl`` / ``val_index.jsonl``.

    Returns:
        List of parsed :class:`~aria_nbv.data_handling.cache_contracts.OracleRriCacheEntry`.
    """
    if not index_path.exists():
        return []
    entries: list[OracleRriCacheEntry] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        entries.append(
            OracleRriCacheEntry(
                key=str(payload.get("key", "")),
                scene_id=str(payload.get("scene_id", "")),
                snippet_id=str(payload.get("snippet_id", "")),
                path=str(payload.get("path", "")),
            ),
        )
    return entries


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
        Set of sample keys (prefix before the first ``.`` in the member name).
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
    """Scan a list of tar shards and return available snippet ids per scene.

    Args:
        tar_paths: Iterable of tar shard paths.
        snippet_key_filter: Optional filter tokens applied to sample keys. Matching uses
            the same semantics as :class:`~aria_nbv.data_handling.efm_dataset.AseEfmDataset`:
            ``key == token or key.endswith(token)`` for any token.
        progress_cb: Optional callback invoked as ``progress_cb(done, total)``.

    Returns:
        Mapping ``scene_id -> {snippet_id, ...}`` where ``snippet_id`` is the sample key.
    """
    tokens = {str(token) for token in (snippet_key_filter or []) if str(token)}

    def _matches_filter(sample_key: str) -> bool:
        if not tokens:
            return True
        return any(sample_key == token or sample_key.endswith(token) for token in tokens)

    out: dict[str, set[str]] = {}
    tar_list = list(tar_paths)
    total = len(tar_list)
    for idx, tar_path in enumerate(tar_list):
        scene_from_path = tar_path.parent.name if tar_path.parent.name.isdigit() else None
        keys = scan_tar_sample_keys(tar_path)
        filtered = [key for key in keys if _matches_filter(key)]
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
    dataset_snippets: int
    cache_train_snippets: int
    cache_val_snippets: int
    cache_all_snippets: int
    coverage_train: float | None
    coverage_val: float | None
    coverage_all: float | None


@dataclass(slots=True)
class CacheCoverageReport:
    """Aggregated coverage report between dataset shards and cache indices."""

    dataset_scenes: int
    dataset_snippets: int
    cache_train_scenes: int
    cache_train_snippets: int
    cache_val_scenes: int
    cache_val_snippets: int
    cache_all_scenes: int
    cache_all_snippets: int
    cache_outside_dataset: int
    per_scene: list[SceneCoverage]

    def as_rows(self) -> list[dict[str, Any]]:
        """Render per-scene coverage as a list of dicts for DataFrame display."""
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
        dataset_snippets: Mapping ``scene_id -> {snippet_id, ...}`` from shard scanning.
        cache_train_snippets: Mapping for train cache split, or ``None`` to treat as empty.
        cache_val_snippets: Mapping for val cache split, or ``None`` to treat as empty.

    Returns:
        Aggregated :class:`CacheCoverageReport`.
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

    def _nonzero_scenes(mapping: Mapping[str, set[str]]) -> int:
        return sum(1 for snippets in mapping.values() if snippets)

    dataset_scenes = _nonzero_scenes(dataset_snippets)
    dataset_snippets_total = sum(len(snippets) for snippets in dataset_snippets.values())
    cache_train_scenes = _nonzero_scenes(train)
    cache_val_scenes = _nonzero_scenes(val)
    cache_all_scenes = _nonzero_scenes(all_cache)

    cache_train_snippets_total = sum(len(snippets) for snippets in train.values())
    cache_val_snippets_total = sum(len(snippets) for snippets in val.values())
    cache_all_snippets_total = sum(len(snippets) for snippets in all_cache.values())

    return CacheCoverageReport(
        dataset_scenes=dataset_scenes,
        dataset_snippets=dataset_snippets_total,
        cache_train_scenes=cache_train_scenes,
        cache_train_snippets=cache_train_snippets_total,
        cache_val_scenes=cache_val_scenes,
        cache_val_snippets=cache_val_snippets_total,
        cache_all_scenes=cache_all_scenes,
        cache_all_snippets=cache_all_snippets_total,
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

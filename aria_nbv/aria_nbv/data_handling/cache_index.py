"""Shared JSONL index and split-management helpers for v2 caches.

This module centralizes the small, file-oriented operations used by both cache
readers and writers:
- reading and writing JSONL index files,
- validating train/val split files against a base oracle index,
- repairing split assignments while preserving prior membership,
- rebuilding oracle indices by scanning payload filenames,
- reading and writing small JSON metadata payloads.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from ..utils import Console
from .cache_contracts import OracleRriCacheEntry, VinSnippetCacheEntry

IndexEntry = TypeVar("IndexEntry", OracleRriCacheEntry, VinSnippetCacheEntry)


def read_index(
    path: Path,
    *,
    entry_type: type[IndexEntry],
    allow_missing: bool = False,
) -> list[IndexEntry]:
    """Read a JSONL index file into dataclass entries."""
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing cache index at {path}")
    entries: list[IndexEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        entries.append(
            entry_type(
                key=item["key"],
                scene_id=item["scene_id"],
                snippet_id=item["snippet_id"],
                path=item["path"],
            ),
        )
    return entries


def serialize_index(entries: Iterable[IndexEntry]) -> str:
    """Serialize index entries to JSONL."""
    rows = list(entries)
    if not rows:
        return ""
    return "\n".join(json.dumps(asdict(entry)) for entry in rows) + "\n"


def write_index(path: Path, entries: Iterable[IndexEntry]) -> None:
    """Write a JSONL index file."""
    path.write_text(serialize_index(entries), encoding="utf-8")


def write_index_if_changed(path: Path, entries: Iterable[IndexEntry]) -> None:
    """Write a JSONL index file only when its payload changes."""
    payload = serialize_index(entries)
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.write_text(payload, encoding="utf-8")


def read_pairs(path: Path, *, allow_missing: bool = False) -> set[tuple[str, str]]:
    """Read ``(scene_id, snippet_id)`` pairs from a JSONL cache index."""
    entries = read_index(path, entry_type=VinSnippetCacheEntry, allow_missing=allow_missing)
    return {(entry.scene_id, entry.snippet_id) for entry in entries}


def validate_oracle_split_indices(
    *,
    base_entries: list[OracleRriCacheEntry],
    train_index_path: Path,
    val_index_path: Path,
) -> tuple[list[OracleRriCacheEntry], list[OracleRriCacheEntry]]:
    """Validate train/val split files against the base oracle index."""
    if not train_index_path.exists() or not val_index_path.exists():
        raise FileNotFoundError(
            "Missing `train_index.jsonl` or `val_index.jsonl`; run "
            "`repair_oracle_cache_indices(...)` or rebuild the cache writer output.",
        )

    train_entries = read_index(train_index_path, entry_type=OracleRriCacheEntry)
    val_entries = read_index(val_index_path, entry_type=OracleRriCacheEntry)
    base_by_key = {entry.key: entry for entry in base_entries}

    train_keys = [entry.key for entry in train_entries]
    val_keys = [entry.key for entry in val_entries]
    if len(set(train_keys)) != len(train_keys) or len(set(val_keys)) != len(val_keys):
        raise ValueError(
            "Duplicate keys in oracle train/val split indices; run `repair_oracle_cache_indices(...)`.",
        )
    if set(train_keys) & set(val_keys):
        raise ValueError(
            "Overlapping keys across oracle train/val split indices; run `repair_oracle_cache_indices(...)`.",
        )
    if any(key not in base_by_key for key in train_keys + val_keys):
        raise ValueError(
            "Oracle train/val split indices contain keys not present in `index.jsonl`; "
            "run `repair_oracle_cache_indices(...)`.",
        )

    base_keys = {entry.key for entry in base_entries}
    split_keys = set(train_keys) | set(val_keys)
    if split_keys != base_keys:
        raise ValueError(
            "Oracle train/val split indices are stale relative to `index.jsonl`; "
            "run `repair_oracle_cache_indices(...)`.",
        )

    return [base_by_key[key] for key in train_keys], [base_by_key[key] for key in val_keys]


def repair_oracle_split_indices(
    *,
    base_entries: list[OracleRriCacheEntry],
    train_index_path: Path,
    val_index_path: Path,
    val_fraction: float,
    console: Console | None = None,
) -> tuple[list[OracleRriCacheEntry], list[OracleRriCacheEntry]]:
    """Repair oracle train/val split files while preserving prior assignments."""
    if not base_entries:
        write_index_if_changed(train_index_path, [])
        write_index_if_changed(val_index_path, [])
        return [], []

    val_fraction = max(0.0, min(1.0, float(val_fraction)))
    if val_fraction <= 0.0:
        write_index_if_changed(train_index_path, base_entries)
        write_index_if_changed(val_index_path, [])
        return list(base_entries), []
    if val_fraction >= 1.0:
        write_index_if_changed(train_index_path, [])
        write_index_if_changed(val_index_path, base_entries)
        return [], list(base_entries)

    base_by_key = {entry.key: entry for entry in base_entries}
    train_prior = read_index(train_index_path, entry_type=OracleRriCacheEntry, allow_missing=True)
    val_prior = read_index(val_index_path, entry_type=OracleRriCacheEntry, allow_missing=True)

    train_entries: list[OracleRriCacheEntry] = []
    val_entries: list[OracleRriCacheEntry] = []
    used_keys: set[str] = set()
    for entry in train_prior:
        if entry.key in base_by_key and entry.key not in used_keys:
            train_entries.append(base_by_key[entry.key])
            used_keys.add(entry.key)
    for entry in val_prior:
        if entry.key in base_by_key and entry.key not in used_keys:
            val_entries.append(base_by_key[entry.key])
            used_keys.add(entry.key)

    target_val = int(round(len(base_entries) * val_fraction))
    if val_entries and len(val_entries) != target_val and console is not None:
        console.warn(
            "Existing oracle val split size differs from `train_val_split`; preserving prior assignments.",
        )

    missing_entries = [entry for entry in base_entries if entry.key not in used_keys]
    for entry in missing_entries:
        if len(val_entries) < target_val:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    write_index_if_changed(train_index_path, train_entries)
    write_index_if_changed(val_index_path, val_entries)
    return train_entries, val_entries


def rebuild_oracle_entries_from_samples(
    *,
    samples_dir: Path,
    cache_dir: Path,
    rng_seed: int | None = None,
) -> list[OracleRriCacheEntry]:
    """Rebuild oracle base index entries from cached sample filenames."""
    entries: list[OracleRriCacheEntry] = []
    for sample_path in sorted(samples_dir.glob("*.pt")):
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
    if rng_seed is not None:
        rng = random.Random(rng_seed)
        rng.shuffle(entries)
    return entries


def load_json_metadata(path: Path) -> dict[str, Any]:
    """Read a JSON metadata file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON metadata."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "load_json_metadata",
    "read_index",
    "read_pairs",
    "rebuild_oracle_entries_from_samples",
    "repair_oracle_split_indices",
    "serialize_index",
    "validate_oracle_split_indices",
    "write_index",
    "write_index_if_changed",
    "write_json_metadata",
]

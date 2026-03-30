"""Shared shard and snippet-selection helpers for EFM-backed datasets.

This module owns the small filesystem and tar-introspection utilities that are
used to resolve requested snippet IDs into concrete shard paths, match snippet
tokens consistently, and expand tar URL patterns into stable path lists.
"""

from __future__ import annotations

import glob
import tarfile
from collections.abc import Iterable
from pathlib import Path

from ..configs import PathConfig


def matches_snippet_token(prefix: str, token: str) -> bool:
    """Return whether a shard member prefix matches a requested snippet token."""

    return prefix == token or prefix.endswith(token)


def looks_like_shard_id(snippet_id: str) -> bool:
    """Return whether a snippet identifier refers to a shard tar."""

    stem = Path(snippet_id).name
    return stem.startswith("shards-") or stem.endswith("_tar") or stem.endswith(".tar")


def split_snippet_ids(snippet_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split mixed snippet identifiers into shard IDs and sample keys."""

    shard_ids: list[str] = []
    sample_keys: list[str] = []
    for snippet_id in snippet_ids:
        if looks_like_shard_id(snippet_id):
            shard_ids.append(snippet_id)
        else:
            sample_keys.append(snippet_id)
    return shard_ids, sample_keys


def expand_tar_paths(tar_urls: Iterable[str | Path]) -> list[Path]:
    """Expand tar URL entries into unique concrete paths.

    Args:
        tar_urls: Iterable of shard paths or glob patterns, absolute or relative.

    Returns:
        Unique resolved paths in stable order.
    """

    expanded: list[Path] = []
    for url in tar_urls:
        url_text = str(url).strip()
        if not url_text:
            continue
        if any(ch in url_text for ch in "*?[]"):
            expanded.extend(Path(match) for match in sorted(glob.glob(url_text)))
        else:
            expanded.append(Path(url_text))

    ordered_unique: list[Path] = []
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
        ordered_unique.append(resolved)
    return ordered_unique


def resolve_tar_from_path(
    *,
    snippet_id: str,
    paths: PathConfig,
) -> Path | None:
    """Resolve an explicit shard path or relative shard reference."""

    candidate = Path(snippet_id)
    if candidate.is_absolute():
        if candidate.suffix != ".tar":
            candidate = candidate.with_suffix(".tar")
        return candidate if candidate.exists() else None
    if candidate.parent != Path():
        resolved = paths.resolve_under_root(candidate)
        if resolved.suffix != ".tar":
            resolved = resolved.with_suffix(".tar")
        return resolved if resolved.exists() else None
    return None


def resolve_tars_for_shard(
    *,
    shard_id: str,
    scene_dirs: list[Path],
) -> list[Path]:
    """Resolve a shard identifier against the configured scene directories."""

    stem = Path(shard_id).stem.removesuffix("_tar")
    matches: list[Path] = []
    for scene_dir in scene_dirs:
        candidate = scene_dir / f"{stem}.tar"
        if candidate.exists():
            matches.append(candidate)
    return matches


def find_tar_for_sample(
    *,
    sample_key: str,
    scene_dirs: list[Path],
) -> Path | None:
    """Find the shard tar that contains a requested sample key."""

    for scene_dir in scene_dirs:
        for tar_path in sorted(scene_dir.glob("*.tar")):
            if _tar_contains_snippet(tar_path, sample_key):
                return tar_path
    return None


def _tar_contains_snippet(tar_path: Path, snippet_token: str) -> bool:
    """Return whether a shard tar contains the requested snippet token."""

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            prefix = member.name.split(".", 1)[0]
            if matches_snippet_token(prefix, snippet_token):
                return True
    return False


__all__ = [
    "expand_tar_paths",
    "find_tar_for_sample",
    "looks_like_shard_id",
    "matches_snippet_token",
    "resolve_tar_from_path",
    "resolve_tars_for_shard",
    "split_snippet_ids",
]

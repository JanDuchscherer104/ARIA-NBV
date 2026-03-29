"""Tests for offline cache coverage utilities."""

from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING

from aria_nbv.data.offline_cache_coverage import (
    compute_cache_coverage,
    expand_tar_urls,
    read_cache_index_entries,
    scan_dataset_snippets,
    scan_tar_sample_keys,
    snippets_by_scene,
)
from aria_nbv.data.offline_cache_types import OracleRriCacheEntry

if TYPE_CHECKING:
    from pathlib import Path


def _write_tar_with_samples(tar_path: Path, sample_keys: list[str]) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for key in sample_keys:
            payload = f"{key}.dummy.txt"
            tmp = tar_path.parent / payload
            tmp.write_text("x", encoding="utf-8")
            tar.add(tmp, arcname=payload)
            tmp.unlink()


def _write_index(path: Path, entries: list[OracleRriCacheEntry]) -> None:
    payload = "\n".join(
        (
            "{"
            f'"key": "{entry.key}", '
            f'"scene_id": "{entry.scene_id}", '
            f'"snippet_id": "{entry.snippet_id}", '
            f'"path": "{entry.path}"'
            "}"
        )
        for entry in entries
    )
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def test_scan_tar_sample_keys(tmp_path: Path) -> None:
    """Extract unique sample keys from a tar shard."""
    tar_path = tmp_path / "81286" / "shards-0000.tar"
    keys = [
        "AriaSyntheticEnvironment_81286_AtekDataSample_000000",
        "AriaSyntheticEnvironment_81286_AtekDataSample_000001",
    ]
    _write_tar_with_samples(tar_path, keys)
    assert scan_tar_sample_keys(tar_path) == set(keys)  # noqa: S101


def test_expand_tar_urls_supports_absolute_globs(tmp_path: Path) -> None:
    """Expand absolute glob patterns into concrete shard paths."""
    tar_path = tmp_path / "100" / "shards-0000.tar"
    _write_tar_with_samples(
        tar_path,
        ["AriaSyntheticEnvironment_100_AtekDataSample_000000"],
    )
    pattern = str(tmp_path / "*" / "*.tar")
    expanded = expand_tar_urls([pattern])
    assert tar_path.resolve() in expanded  # noqa: S101


def test_compute_cache_coverage_from_synthetic_shards(tmp_path: Path) -> None:
    """Compute coverage report for synthetic dataset shards + cache indices."""
    tar_scene_100 = tmp_path / "100" / "shards-0000.tar"
    tar_scene_200 = tmp_path / "200" / "shards-0000.tar"
    dataset_scene_100 = [
        "AriaSyntheticEnvironment_100_AtekDataSample_000000",
        "AriaSyntheticEnvironment_100_AtekDataSample_000001",
    ]
    dataset_scene_200 = ["AriaSyntheticEnvironment_200_AtekDataSample_000000"]
    _write_tar_with_samples(tar_scene_100, dataset_scene_100)
    _write_tar_with_samples(tar_scene_200, dataset_scene_200)

    dataset_snippets = scan_dataset_snippets([tar_scene_100, tar_scene_200])

    train_entries = [
        OracleRriCacheEntry(
            key="k0",
            scene_id="100",
            snippet_id=dataset_scene_100[0],
            path="samples/k0.pt",
        ),
        OracleRriCacheEntry(
            key="k1",
            scene_id="300",
            snippet_id="AriaSyntheticEnvironment_300_AtekDataSample_000000",
            path="samples/k1.pt",
        ),
    ]
    val_entries = [
        OracleRriCacheEntry(
            key="k2",
            scene_id="200",
            snippet_id=dataset_scene_200[0],
            path="samples/k2.pt",
        ),
    ]

    report = compute_cache_coverage(
        dataset_snippets=dataset_snippets,
        cache_train_snippets=snippets_by_scene(train_entries),
        cache_val_snippets=snippets_by_scene(val_entries),
    )

    expected_dataset_scenes = 2
    expected_dataset_snippets = 3
    expected_cache_all_snippets = 3
    expected_cache_outside_dataset = 1
    assert report.dataset_scenes == expected_dataset_scenes  # noqa: S101
    assert report.dataset_snippets == expected_dataset_snippets  # noqa: S101
    assert report.cache_all_snippets == expected_cache_all_snippets  # noqa: S101
    assert report.cache_outside_dataset == expected_cache_outside_dataset  # noqa: S101

    per_scene = {row.scene_id: row for row in report.per_scene}
    expected_scene_100_coverage = 0.5
    expected_scene_200_coverage = 1.0
    assert per_scene["100"].coverage_all == expected_scene_100_coverage  # noqa: S101
    assert per_scene["200"].coverage_all == expected_scene_200_coverage  # noqa: S101


def test_read_cache_index_entries_roundtrip(tmp_path: Path) -> None:
    """Parse JSONL cache entries and group by scene."""
    index_path = tmp_path / "index.jsonl"
    entries = [
        OracleRriCacheEntry(
            key="k0",
            scene_id="100",
            snippet_id="s0",
            path="samples/k0.pt",
        ),
        OracleRriCacheEntry(
            key="k1",
            scene_id="100",
            snippet_id="s1",
            path="samples/k1.pt",
        ),
    ]
    _write_index(index_path, entries)
    loaded = read_cache_index_entries(index_path)
    grouped = snippets_by_scene(loaded)
    assert grouped == {"100": {"s0", "s1"}}  # noqa: S101


def test_scan_dataset_snippets_respects_key_filter(tmp_path: Path) -> None:
    """Apply snippet_key_filter semantics when scanning shards."""
    tar_path = tmp_path / "100" / "shards-0000.tar"
    keys = [
        "AriaSyntheticEnvironment_100_AtekDataSample_000000",
        "AriaSyntheticEnvironment_100_AtekDataSample_000001",
    ]
    _write_tar_with_samples(tar_path, keys)

    filtered = scan_dataset_snippets([tar_path], snippet_key_filter=["000001"])
    assert filtered == {"100": {keys[1]}}  # noqa: S101

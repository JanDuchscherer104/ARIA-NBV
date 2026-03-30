"""Integration test for VIN snippet cache on real data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aria_nbv.data_handling import (
    VinSnippetCacheConfig,
    VinSnippetCacheDatasetConfig,
    VinSnippetCacheWriterConfig,
)
from aria_nbv.data_handling.oracle_cache import OracleRriCacheConfig


def _skip_if_missing_real_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        pytest.skip("Missing offline oracle cache directory.")
    index_path = cache_dir / "index.jsonl"
    if not index_path.exists():
        pytest.skip("Missing offline cache index.")
    if not (cache_dir / "metadata.json").exists():
        pytest.skip("Missing offline cache metadata.")
    if not (cache_dir / "samples").exists():
        pytest.skip("Missing offline cache samples.")


def _first_entry(index_path: Path) -> dict[str, str] | None:
    lines = index_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.strip():
            return json.loads(line)
    return None


def test_vin_snippet_cache_real_data(tmp_path: Path) -> None:
    """Build and load a VIN snippet cache using a real offline cache entry."""
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / ".data" / "oracle_rri_cache"
    _skip_if_missing_real_cache(cache_dir)

    entry = _first_entry(cache_dir / "index.jsonl")
    if entry is None:
        pytest.skip("Offline cache index is empty.")

    scene_id = entry.get("scene_id", "")
    if not scene_id:
        pytest.skip("Offline cache entry missing scene_id.")

    data_dir = repo_root / ".data" / "ase_efm" / scene_id
    if not data_dir.exists():
        pytest.skip("Missing ASE shard directory for scene.")
    if not any(data_dir.glob("*.tar")):
        pytest.skip("Missing ASE tar shards for scene.")

    vin_cache_dir = tmp_path / "vin_snippets"
    writer_cfg = VinSnippetCacheWriterConfig(
        cache=VinSnippetCacheConfig(cache_dir=vin_cache_dir),
        source_cache=OracleRriCacheConfig(cache_dir=cache_dir),
        max_samples=1,
        overwrite=True,
        map_location="cpu",
    )
    writer_cfg.setup_target().run()

    ds_cfg = VinSnippetCacheDatasetConfig(cache=writer_cfg.cache, map_location="cpu")
    dataset = ds_cfg.setup_target()
    sample = dataset[0]
    assert sample.points_world.shape[-1] == 4
    assert sample.t_world_rig.tensor().shape[-1] == 12

"""Tests for skipping duplicates in offline cache generation."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from oracle_rri.data.offline_cache_types import OracleRriCacheEntry


@dataclass(slots=True)
class DummySample:
    """Minimal sample stub exposing scene/snippet identifiers."""

    scene_id: str
    snippet_id: str


def test_cache_writer_skips_existing_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip samples that already exist in the cache index."""
    if importlib.util.find_spec("power_spherical") is None:
        pytest.skip("Missing power_spherical dependency")
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    labeler_mod = importlib.import_module("oracle_rri.pipelines.oracle_rri_labeler")
    utils_mod = importlib.import_module("oracle_rri.utils")

    dataset = [
        DummySample("scene_1", "snippet_1"),
        DummySample("scene_1", "snippet_1"),
        DummySample("scene_2", "snippet_2"),
    ]

    def _fake_setup_target(_self: object) -> list[DummySample]:
        return dataset

    def _fake_labeler_setup(_self: object) -> object:
        return object()

    def _fake_write_sample(
        self: object,
        sample: DummySample,
        samples_dir: Path,
    ) -> OracleRriCacheEntry:
        key = f"{sample.scene_id}_{sample.snippet_id}_test"
        sample_path = samples_dir / f"{key}.pt"
        sample_path.write_text("x", encoding="utf-8")
        return offline_mod.OracleRriCacheEntry(
            key=key,
            scene_id=sample.scene_id,
            snippet_id=sample.snippet_id,
            path=str(sample_path.relative_to(self.config.cache.cache_dir)),
        )

    monkeypatch.setattr(
        offline_mod.AseEfmDatasetConfig,
        "setup_target",
        _fake_setup_target,
        raising=True,
    )
    monkeypatch.setattr(
        labeler_mod.OracleRriLabelerConfig,
        "setup_target",
        _fake_labeler_setup,
        raising=True,
    )
    monkeypatch.setattr(
        offline_mod.OracleRriCacheWriter,
        "_write_sample",
        _fake_write_sample,
        raising=True,
    )

    cache_cfg = offline_mod.OracleRriCacheConfig(cache_dir=tmp_path / "cache")
    cache_cfg.cache_dir.mkdir(parents=True)
    cache_cfg.samples_dir.mkdir(parents=True)

    existing_entry = offline_mod.OracleRriCacheEntry(
        key="existing_key",
        scene_id="scene_1",
        snippet_id="snippet_1",
        path="samples/existing_key.pt",
    )
    (cache_cfg.cache_dir / existing_entry.path).write_text("x", encoding="utf-8")
    cache_cfg.index_path.write_text(
        json.dumps(asdict(existing_entry)) + "\n",
        encoding="utf-8",
    )

    writer_cfg = offline_mod.OracleRriCacheWriterConfig(
        cache=cache_cfg,
        include_backbone=False,
        backbone=None,
        overwrite=True,
        verbosity=utils_mod.Verbosity.QUIET,
    )

    entries = writer_cfg.setup_target().run()
    assert len(entries) == 1  # noqa: S101
    assert entries[0].scene_id == "scene_2"  # noqa: S101

    index_entries = [
        json.loads(line)
        for line in cache_cfg.index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index_pairs = {(entry["scene_id"], entry["snippet_id"]) for entry in index_entries}
    assert index_pairs == {("scene_1", "snippet_1"), ("scene_2", "snippet_2")}  # noqa: S101

    expected_total = 2
    meta_payload = json.loads(cache_cfg.metadata_path.read_text(encoding="utf-8"))
    assert meta_payload["num_samples"] == expected_total  # noqa: S101

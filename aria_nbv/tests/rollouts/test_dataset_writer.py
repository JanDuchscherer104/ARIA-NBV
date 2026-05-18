"""Tests for rollout dataset writer lineage helpers."""

# ruff: noqa: S101

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest

from aria_nbv.rollouts.dataset_writer import RolloutDatasetWriter, _RolloutSourceLineageBuilder
from aria_nbv.rollouts.manifest import RolloutStoreManifestContext
from aria_nbv.rollouts.shard_manifest import RolloutShardEntry, canonical_rollout_shard_id, write_rollout_shard_manifest
from aria_nbv.rollouts.shards import plan_rollout_shards, run_rollout_shard, summarize_rollout_shard_campaign
from aria_nbv.rollouts.zarr_store import write_rollout_zarr_store
from tests.rollout_fixtures import build_rollout_records


class _FakeManifest(msgspec.Struct):
    version: int = 7


class _FakeSource:
    def __init__(self, dataset: _FakeDataset, *, store_dir: Path) -> None:
        self._dataset = dataset
        self.store = SimpleNamespace(store_dir=store_dir)

    def setup_target(self) -> "_FakeDataset":
        return self._dataset


class _FakeDataset:
    def __init__(self, records: list[Any], *, config_split: str = "train") -> None:
        self.manifest = _FakeManifest()
        self.config = SimpleNamespace(split=config_split)
        self._records = records
        self._record_by_pair = {(record.scene_id, record.snippet_id): record for record in records}

    def __len__(self) -> int:
        return len(self._records)


class _FakeRolloutConfig:
    def __init__(self, records: list[Any], *, store_dir: Path, source_split: str = "train") -> None:
        self.source = _FakeSource(_FakeDataset(records, config_split=source_split), store_dir=store_dir / "vin_offline")
        self.store = SimpleNamespace(store_dir=store_dir / "configured-rollouts.zarr")
        self._dump_token = "fake-rollout-config-v1"

    def model_dump_jsonable(self) -> dict[str, Any]:
        return {"dump_token": self._dump_token, "source_store": self.source.store.store_dir.as_posix()}

    def model_copy(self, *, deep: bool = False) -> "_FakeRolloutConfig":
        return deepcopy(self) if deep else self

    def setup_target(self) -> "_FakeShardWriter":
        return _FakeShardWriter(self)


class _FakeShardWriter:
    def __init__(self, config: _FakeRolloutConfig) -> None:
        self.config = config

    def run(self, *, invocation: object | None = None, shard_entry: RolloutShardEntry | None = None):
        del invocation
        if shard_entry is None:
            raise AssertionError("Shard writer tests must pass a shard entry.")
        records = build_rollout_records(horizon=1, num_samples=6, seed=33)[:1]
        records[0].lineage.source_offline_store_manifest_hash = shard_entry.source_manifest_hash
        records[0].lineage.source_cache_version = shard_entry.source_cache_version
        records[0].lineage.split_manifest_hash = shard_entry.split_manifest_hash
        return write_rollout_zarr_store(
            self.config.store.store_dir,
            records,
            source_offline_store_version=shard_entry.source_cache_version,
            split_manifest_hash=shard_entry.split_manifest_hash,
            manifest_context=RolloutStoreManifestContext(shard=shard_entry.to_jsonable()),
        )


def test_split_manifest_hash_tracks_source_rows_and_order() -> None:
    rows = [
        {
            "order": 0,
            "sample_index": 1,
            "sample_key": "a",
            "scene_id": "scene-a",
            "snippet_id": "snippet-a",
            "split": "train",
            "source_shard_id": "shard-0",
            "source_shard_row": 0,
        },
        {
            "order": 1,
            "sample_index": 2,
            "sample_key": "b",
            "scene_id": "scene-b",
            "snippet_id": "snippet-b",
            "split": "train",
            "source_shard_id": "shard-0",
            "source_shard_row": 1,
        },
    ]

    base = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="source", split="train", records=rows
    )
    reordered = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="source", split="train", records=list(reversed(rows))
    )
    changed_source = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="other", split="train", records=rows
    )

    assert base != reordered
    assert base != changed_source


def test_rollout_shard_manifest_planning_is_deterministic_and_order_sensitive(tmp_path: Path) -> None:
    records = [_fake_record(index) for index in range(4)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)

    first = plan_rollout_shards(config, rows_per_shard=2)
    second = plan_rollout_shards(config, rows_per_shard=2)
    reversed_entries = plan_rollout_shards(
        _FakeRolloutConfig(list(reversed(records)), store_dir=tmp_path), rows_per_shard=2
    )

    assert [entry.to_jsonable() for entry in first] == [entry.to_jsonable() for entry in second]
    assert [entry.shard_id for entry in first] == ["shard-000000", "shard-000001"]
    assert first[0].split_manifest_hash != reversed_entries[0].split_manifest_hash
    assert first[0].rows[0].source_shard_id == "vin-shard-000000"
    assert first[0].rows[0].source_shard_row == 0


def test_rollout_shard_id_canonicalization_accepts_padded_and_unpadded_forms() -> None:
    assert canonical_rollout_shard_id(7) == "shard-000007"
    assert canonical_rollout_shard_id("7") == "shard-000007"
    assert canonical_rollout_shard_id("shard-7") == "shard-000007"
    assert canonical_rollout_shard_id("shard-000007") == "shard-000007"


def test_rollout_shard_mode_rejects_manifest_source_row_mismatch(tmp_path: Path) -> None:
    records = [_fake_record(0)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)
    entry = plan_rollout_shards(config, rows_per_shard=1)[0]
    bad_entry = replace(entry, rows=(replace(entry.rows[0], scene_id="other-scene"),))
    writer = RolloutDatasetWriter.__new__(RolloutDatasetWriter)

    with pytest.raises(ValueError, match="does not match"):
        writer._apply_shard_manifest(config.source.setup_target(), bad_entry)


def test_rollout_shard_lineage_uses_row_split_when_source_config_exposes_all(tmp_path: Path) -> None:
    records = [_fake_record(0)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path, source_split="all")
    entry = plan_rollout_shards(config, rows_per_shard=1)[0]
    dataset = config.source.setup_target()
    writer = RolloutDatasetWriter.__new__(RolloutDatasetWriter)

    writer._apply_shard_manifest(dataset, entry)
    source_lineage = _RolloutSourceLineageBuilder.from_dataset(dataset, max_samples=len(dataset))

    RolloutDatasetWriter._validate_shard_lineage(source_lineage, entry)


def test_rollout_shard_atomic_promotion_writes_markers_and_skips_completed(tmp_path: Path) -> None:
    records = [_fake_record(0)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)
    entry = plan_rollout_shards(config, rows_per_shard=1)[0]
    tmp_dir = tmp_path / "tmp" / "shard-000000.tmp"
    final_dir = tmp_path / "final" / "shard-000000"

    result = run_rollout_shard(config, shard_entry=entry, output_tmp=tmp_dir, output_final=final_dir)
    skipped = run_rollout_shard(config, shard_entry=entry, output_tmp=tmp_path / "other.tmp", output_final=final_dir)

    assert not result.skipped
    assert result.success_path.exists()
    assert result.owner_path.exists()
    assert not tmp_dir.exists()
    assert skipped.skipped


def test_rollout_shard_resume_rejects_tampered_owner_sidecar(tmp_path: Path) -> None:
    records = [_fake_record(0)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)
    entry = plan_rollout_shards(config, rows_per_shard=1)[0]
    final_dir = tmp_path / "final" / "shard-000000"

    result = run_rollout_shard(
        config,
        shard_entry=entry,
        output_tmp=tmp_path / "tmp" / "shard-000000.tmp",
        output_final=final_dir,
    )
    owner_payload = msgspec.json.decode(result.owner_path.read_bytes())
    owner_payload["num_source_rows"] = 999
    result.owner_path.write_bytes(msgspec.json.encode(owner_payload))

    with pytest.raises(RuntimeError, match="not a validated completed shard"):
        run_rollout_shard(
            config,
            shard_entry=entry,
            output_tmp=tmp_path / "tmp" / "retry.tmp",
            output_final=final_dir,
        )


def test_rollout_shard_atomic_promotion_rejects_stale_paths(tmp_path: Path) -> None:
    records = [_fake_record(0)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)
    entry = plan_rollout_shards(config, rows_per_shard=1)[0]
    stale_tmp = tmp_path / "tmp" / "shard-000000.tmp"
    stale_tmp.mkdir(parents=True)
    partial_final = tmp_path / "final" / "shard-000000"
    partial_final.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Temporary rollout shard path already exists"):
        run_rollout_shard(config, shard_entry=entry, output_tmp=stale_tmp, output_final=tmp_path / "new-final")
    with pytest.raises(RuntimeError, match="Final rollout shard path exists"):
        run_rollout_shard(config, shard_entry=entry, output_tmp=tmp_path / "fresh.tmp", output_final=partial_final)


def test_rollout_shard_campaign_status_reports_retry_classes(tmp_path: Path) -> None:
    records = [_fake_record(index) for index in range(4)]
    config = _FakeRolloutConfig(records, store_dir=tmp_path)
    entries = plan_rollout_shards(config, rows_per_shard=1)
    manifest_path = tmp_path / "rollout_shards.jsonl"
    final_root = tmp_path / "final"
    write_rollout_shard_manifest(manifest_path, entries)

    run_rollout_shard(
        config,
        shard_entry=entries[0],
        output_tmp=tmp_path / "tmp" / "shard-000000.tmp",
        output_final=final_root / "shard-000000",
    )
    (final_root / "_FAILED.shard-000001.2026-05-15T00-00-00Z.json").write_text(
        '{"error": "synthetic failure"}',
        encoding="utf-8",
    )
    (final_root / "shard-000002").mkdir(parents=True)

    campaign = summarize_rollout_shard_campaign(manifest_path, final_root=final_root)
    by_id = {shard.shard_id: shard for shard in campaign.shards}

    assert campaign.counts == {"succeeded": 1, "failed": 1, "incomplete": 1, "missing": 1}
    assert by_id["shard-000000"].status == "succeeded"
    assert by_id["shard-000001"].status == "failed"
    assert by_id["shard-000001"].failed_markers
    assert by_id["shard-000002"].status == "incomplete"
    assert by_id["shard-000003"].status == "missing"


def _fake_record(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        sample_index=index,
        sample_key=f"scene-a:snippet-{index:03d}",
        scene_id="scene-a",
        snippet_id=f"snippet-{index:03d}",
        split="train",
        shard_id="vin-shard-000000",
        row=index,
    )

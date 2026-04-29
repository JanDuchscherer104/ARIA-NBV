"""Focused round-trip tests for the immutable VIN offline dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from aria_nbv.data_handling import (
    OFFLINE_DATASET_VERSION,
    VinOfflineDatasetConfig,
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
    VinOfflineSourceConfig,
    VinOfflineStoreConfig,
    VinOracleBatch,
    VinSnippetView,
    collect_vin_offline_dataset_stats,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from aria_nbv.data_handling._offline_format import VinOfflineBlockSpec
from aria_nbv.data_handling._offline_store import VinOfflineStoreReader
from aria_nbv.data_handling._offline_writer import _assign_splits
from aria_nbv.lightning.lit_datamodule import VinDataModuleConfig
from aria_nbv.rendering.candidate_depth_renderer import CandidateDepths
from aria_nbv.rri_metrics.types import RriResult
from aria_nbv.utils import Stage

PoseTW = pytest.importorskip("efm3d.aria.pose").PoseTW
PerspectiveCameras = pytest.importorskip(
    "pytorch3d.renderer.cameras",
).PerspectiveCameras


def _write_sample_index(path: Path, records: list[VinOfflineIndexRecord]) -> None:
    """Write a small sample index without importing internal helpers."""

    payload = "\n".join(json.dumps(asdict(record), sort_keys=True) for record in records)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _read_sample_index_rows(path: Path) -> list[dict[str, object]]:
    """Read the sample index into plain dictionaries for assertions."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_pose_batch(num: int, *, offset: float = 0.0) -> PoseTW:
    rotation = torch.eye(3, dtype=torch.float32).expand(num, 3, 3).clone()
    translation = torch.zeros((num, 3), dtype=torch.float32)
    translation[:, 0] = offset
    translation[:, 1] = torch.arange(num, dtype=torch.float32)
    return PoseTW.from_Rt(rotation, translation)


def _make_stub_depths(num_candidates: int, *, offset: float = 0.0) -> CandidateDepths:
    depths = torch.full((num_candidates, 4, 4), 1.0 + offset, dtype=torch.float32)
    depths_valid = torch.ones_like(depths, dtype=torch.bool)
    poses = _make_pose_batch(num_candidates, offset=offset)
    ref_pose = _make_pose_batch(1, offset=offset).squeeze(0)
    rotation = torch.eye(3, dtype=torch.float32).expand(num_candidates, 3, 3).clone()
    translation = torch.zeros((num_candidates, 3), dtype=torch.float32)
    focal = torch.full((num_candidates, 2), 50.0, dtype=torch.float32)
    principal = torch.full((num_candidates, 2), 2.0, dtype=torch.float32)
    image_size = torch.full((num_candidates, 2), 4.0, dtype=torch.float32)
    p3d = PerspectiveCameras(
        R=rotation,
        T=translation,
        focal_length=focal,
        principal_point=principal,
        image_size=image_size,
        in_ndc=False,
    )
    return CandidateDepths(
        depths=depths,
        depths_valid_mask=depths_valid,
        poses=poses,
        reference_pose=ref_pose,
        candidate_indices=torch.arange(num_candidates, dtype=torch.long),
        camera=None,
        p3d_cameras=p3d,
    )


def _make_stub_rri(num_candidates: int) -> RriResult:
    values = torch.linspace(0.1, 0.1 * num_candidates, num_candidates, dtype=torch.float32)
    return RriResult(
        rri=values,
        pm_dist_before=torch.full((num_candidates,), 0.5, dtype=torch.float32),
        pm_dist_after=torch.full((num_candidates,), 0.4, dtype=torch.float32),
        pm_acc_before=torch.full((num_candidates,), 0.3, dtype=torch.float32),
        pm_comp_before=torch.full((num_candidates,), 0.2, dtype=torch.float32),
        pm_acc_after=torch.full((num_candidates,), 0.25, dtype=torch.float32),
        pm_comp_after=torch.full((num_candidates,), 0.15, dtype=torch.float32),
    )


def _make_vin_snippet(*, offset: float = 0.0) -> VinSnippetView:
    points_world = torch.tensor(
        [
            [offset + 0.0, 0.0, 0.0, 0.1],
            [offset + 1.0, 0.0, 0.0, 0.2],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2], dtype=torch.int64)
    return VinSnippetView(
        points_world=points_world,
        lengths=lengths,
        t_world_rig=_make_pose_batch(2, offset=offset),
    )


def _write_test_store(tmp_path: Path) -> VinOfflineStoreConfig:
    """Create a small immutable VIN offline store for reader tests."""

    store_cfg = VinOfflineStoreConfig(store_dir=tmp_path / "vin_offline")
    store_cfg.store_dir.mkdir(parents=True, exist_ok=True)
    store_cfg.shards_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows = [
        prepare_vin_offline_sample(
            scene_id="scene-a",
            snippet_id="snippet-000",
            vin_snippet=_make_vin_snippet(offset=0.0),
            candidates=None,
            depths=_make_stub_depths(2, offset=0.0),
            rri=_make_stub_rri(2),
            candidate_pcs=None,
            backbone_out=None,
            max_candidates=4,
            include_depths=True,
            include_candidate_pcs=False,
            include_backbone=False,
            sample_key="sample-0",
        ),
        prepare_vin_offline_sample(
            scene_id="scene-b",
            snippet_id="snippet-001",
            vin_snippet=_make_vin_snippet(offset=10.0),
            candidates=None,
            depths=_make_stub_depths(3, offset=10.0),
            rri=_make_stub_rri(3),
            candidate_pcs=None,
            backbone_out=None,
            max_candidates=4,
            include_depths=True,
            include_candidate_pcs=False,
            include_backbone=False,
            sample_key="sample-1",
        ),
        prepare_vin_offline_sample(
            scene_id="scene-c",
            snippet_id="snippet-002",
            vin_snippet=_make_vin_snippet(offset=20.0),
            candidates=None,
            depths=_make_stub_depths(2, offset=20.0),
            rri=_make_stub_rri(2),
            candidate_pcs=None,
            backbone_out=None,
            max_candidates=4,
            include_depths=True,
            include_candidate_pcs=False,
            include_backbone=False,
            sample_key="sample-2",
        ),
    ]

    shard_dir = store_cfg.shards_dir / "shard-000000"
    shard_spec, local_records = flush_prepared_samples_to_shard(
        shard_index=0,
        shard_dir=shard_dir,
        rows=prepared_rows,
    )
    index_records = [
        VinOfflineIndexRecord(
            sample_index=0,
            sample_key=local_records[0].sample_key,
            scene_id=local_records[0].scene_id,
            snippet_id=local_records[0].snippet_id,
            split="train",
            shard_id=local_records[0].shard_id,
            row=local_records[0].row,
        ),
        VinOfflineIndexRecord(
            sample_index=1,
            sample_key=local_records[1].sample_key,
            scene_id=local_records[1].scene_id,
            snippet_id=local_records[1].snippet_id,
            split="train",
            shard_id=local_records[1].shard_id,
            row=local_records[1].row,
        ),
        VinOfflineIndexRecord(
            sample_index=2,
            sample_key=local_records[2].sample_key,
            scene_id=local_records[2].scene_id,
            snippet_id=local_records[2].snippet_id,
            split="val",
            shard_id=local_records[2].shard_id,
            row=local_records[2].row,
        ),
    ]
    manifest = VinOfflineManifest(
        version=OFFLINE_DATASET_VERSION,
        created_at="2026-03-29T00:00:00Z",
        source={"dataset_config": {}},
        oracle={"max_candidates": 4},
        vin={"pad_points": 4},
        materialized_blocks=VinOfflineMaterializedBlocks(
            backbone=False,
            depths=True,
            candidate_pcs=False,
            counterfactuals=False,
        ),
        stats={"num_samples": 3},
        provenance={},
        shards=[shard_spec],
    )
    manifest.write(store_cfg.manifest_path)
    _write_sample_index(store_cfg.sample_index_path, index_records)
    store_cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        store_cfg.split_path("all"),
        torch.tensor([0, 1, 2], dtype=torch.long).numpy(),
        allow_pickle=False,
    )
    np.save(
        store_cfg.split_path("train"),
        torch.tensor([0, 1], dtype=torch.long).numpy(),
        allow_pickle=False,
    )
    np.save(
        store_cfg.split_path("val"),
        torch.tensor([2], dtype=torch.long).numpy(),
        allow_pickle=False,
    )
    return store_cfg


def _supports_worker_tensor_sharing() -> bool:
    """Return whether this host can move worker tensors back to the parent."""

    script = """
import torch
from torch.utils.data import DataLoader, Dataset

class _Dataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32)

torch.multiprocessing.set_sharing_strategy("file_system")
loader = DataLoader(_Dataset(), batch_size=1, num_workers=1, persistent_workers=True)
batch = next(iter(loader))
assert batch.shape == (1, 1)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0


def _make_split_record(sample_key: str, row: int) -> VinOfflineIndexRecord:
    """Build one lightweight index record for split-assignment tests."""

    return VinOfflineIndexRecord(
        sample_index=-1,
        sample_key=sample_key,
        scene_id=f"scene-{sample_key}",
        snippet_id=f"snippet-{sample_key}",
        split="all",
        shard_id="shard-000000",
        row=row,
    )


def test_assign_splits_is_stable_by_sample_key() -> None:
    """Split membership should be stable across input order permutations."""

    records_a = [
        _make_split_record("alpha", 0),
        _make_split_record("beta", 1),
        _make_split_record("gamma", 2),
        _make_split_record("delta", 3),
        _make_split_record("epsilon", 4),
    ]
    records_b = [
        _make_split_record("gamma", 0),
        _make_split_record("alpha", 1),
        _make_split_record("epsilon", 2),
        _make_split_record("delta", 3),
        _make_split_record("beta", 4),
    ]

    splits_a = _assign_splits(records=records_a, val_fraction=0.4)
    splits_b = _assign_splits(records=records_b, val_fraction=0.4)

    val_keys_a = {records_a[int(idx)].sample_key for idx in splits_a["val"]}
    val_keys_b = {records_b[int(idx)].sample_key for idx in splits_b["val"]}
    train_keys_a = {records_a[int(idx)].sample_key for idx in splits_a["train"]}
    train_keys_b = {records_b[int(idx)].sample_key for idx in splits_b["train"]}

    assert val_keys_a == val_keys_b  # noqa: S101
    assert train_keys_a == train_keys_b  # noqa: S101

    val_idx_a = {int(idx) for idx in splits_a["val"]}
    train_idx_a = {int(idx) for idx in splits_a["train"]}
    assert [records_a[int(idx)].sample_key for idx in splits_a["val"]] == [  # noqa: S101
        record.sample_key for idx, record in enumerate(records_a) if idx in val_idx_a
    ]
    assert [records_a[int(idx)].sample_key for idx in splits_a["train"]] == [  # noqa: S101
        record.sample_key for idx, record in enumerate(records_a) if idx in train_idx_a
    ]


def test_collect_vin_offline_dataset_stats_summarizes_store(tmp_path: Path) -> None:
    """Immutable offline diagnostics should summarize coverage and tensor stats."""

    store_cfg = _write_test_store(tmp_path)

    stats = collect_vin_offline_dataset_stats(store_cfg, max_samples=2)

    assert stats.num_samples == 3  # noqa: S101
    assert stats.sampled_samples == 2  # noqa: S101
    assert stats.split_counts == {"train": 2, "val": 1}  # noqa: S101
    assert stats.num_scenes == 3  # noqa: S101
    assert stats.candidate_count.count == 2  # noqa: S101
    assert stats.rri.count == 5  # noqa: S101
    assert stats.vin_points.mean == 2.0  # noqa: S101
    assert stats.numeric_bytes > 0  # noqa: S101


def test_vin_offline_dataset_round_trip(tmp_path: Path) -> None:
    store_cfg = _write_test_store(tmp_path)

    sample_dataset = VinOfflineDatasetConfig(
        store=store_cfg,
        return_format="sample",
        split="all",
    ).setup_target()
    assert len(sample_dataset) == 3  # noqa: S101
    first = sample_dataset[0]
    assert first.scene_id == "scene-a"  # noqa: S101
    assert first.oracle.candidate_count == 2  # noqa: S101
    assert int(first.oracle.rri.shape[0]) == 4  # noqa: S101
    assert torch.isnan(first.oracle.rri[2:]).all()  # noqa: S101
    assert int(first.vin_snippet.lengths[0].item()) == 2  # noqa: S101

    stored_manifest = VinOfflineManifest.read(store_cfg.manifest_path)
    assert stored_manifest.version == OFFLINE_DATASET_VERSION  # noqa: S101
    assert stored_manifest.shards[0].shard_id == "shard-000000"  # noqa: S101
    assert stored_manifest.shards[0].blocks["vin.points_world"].kind == "zarr_array"  # noqa: S101

    sample_index_rows = _read_sample_index_rows(store_cfg.sample_index_path)
    assert sample_index_rows[0]["split"] == "train"  # noqa: S101
    assert sample_index_rows[1]["split"] == "train"  # noqa: S101
    assert sample_index_rows[2]["split"] == "val"  # noqa: S101

    batch_dataset = VinOfflineDatasetConfig(
        store=store_cfg,
        return_format="vin_batch",
        split="train",
    ).setup_target()
    batch = batch_dataset[0]
    assert isinstance(batch, VinOracleBatch)  # noqa: S101
    assert batch.scene_id == "scene-a"  # noqa: S101
    assert int(batch.rri.shape[0]) == 4  # noqa: S101
    assert int(batch.resolved_candidate_count().item()) == 2  # noqa: S101
    assert batch.candidate_valid_mask().tolist() == [True, True, False, False]  # noqa: S101


def test_vin_offline_store_writes_indexed_record_blocks(tmp_path: Path) -> None:
    """Optional record blocks should use indexed payload blobs plus offsets."""

    store_cfg = _write_test_store(tmp_path)
    manifest = VinOfflineManifest.read(store_cfg.manifest_path)
    block = manifest.shards[0].blocks["oracle.depths_payload"]

    assert block.kind == "msgpack_indexed_records"  # noqa: S101
    assert block.paths == [  # noqa: S101
        VinOfflineBlockSpec.msgpack_records_path("oracle.depths_payload"),
        VinOfflineBlockSpec.msgpack_records_offsets_path("oracle.depths_payload"),
    ]
    shard_dir = store_cfg.store_dir / manifest.shards[0].relative_dir
    assert (shard_dir / block.paths[0]).is_file()  # noqa: S101
    assert (shard_dir / block.paths[1]).is_file()  # noqa: S101
    offsets = np.load(shard_dir / block.paths[1], allow_pickle=False)
    assert offsets.tolist()[0] == 0  # noqa: S101
    assert offsets.shape == (4,)  # noqa: S101
    assert np.all(np.diff(offsets) > 0)  # noqa: S101


def test_vin_offline_store_reads_indexed_record_blocks(
    tmp_path: Path,
) -> None:
    """Indexed record blocks should load one row directly from the shard blob."""

    store_cfg = _write_test_store(tmp_path)
    reader = VinOfflineStoreReader(store_cfg)
    record = reader.get_split_records("all")[1]
    payload = reader.read_optional_record(record, "oracle.depths_payload")
    assert payload is not None  # noqa: S101
    decoded = CandidateDepths.from_serializable(payload, device=torch.device("cpu"))
    assert decoded.candidate_indices.tolist() == [0, 1, 2]  # noqa: S101
    assert tuple(decoded.depths.shape) == (3, 4, 4)  # noqa: S101


def test_vin_offline_store_rejects_unsupported_manifest_version(tmp_path: Path) -> None:
    """Runtime readers should only accept the current immutable store version."""

    store_cfg = _write_test_store(tmp_path)
    manifest = VinOfflineManifest.read(store_cfg.manifest_path)
    manifest.version = OFFLINE_DATASET_VERSION - 1
    manifest.write(store_cfg.manifest_path)

    with pytest.raises(ValueError, match="Unsupported VIN offline dataset version"):
        VinOfflineStoreReader(store_cfg)


def test_vin_offline_store_rejects_unsupported_record_block_kind(tmp_path: Path) -> None:
    """Runtime readers should reject unsupported optional-record block encodings."""

    store_cfg = _write_test_store(tmp_path)
    manifest = VinOfflineManifest.read(store_cfg.manifest_path)
    manifest.shards[0].blocks["oracle.depths_payload"].kind = "msgpack_records"
    manifest.write(store_cfg.manifest_path)

    reader = VinOfflineStoreReader(store_cfg)
    record = reader.get_split_records("all")[1]
    with pytest.raises(ValueError, match="Unsupported VIN offline block kind"):
        reader.read_optional_record(record, "oracle.depths_payload")


def test_vin_offline_dataset_vin_batch_skips_optional_record_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIN-batch reads should not decode optional diagnostic record payloads."""

    store_cfg = _write_test_store(tmp_path)
    dataset = VinOfflineDatasetConfig(
        store=store_cfg,
        return_format="vin_batch",
        split="train",
    ).setup_target()

    def _raise_if_called(*_: object, **__: object) -> None:
        raise AssertionError("vin_batch path should not touch optional record blocks")

    monkeypatch.setattr(dataset._store, "read_optional_record", _raise_if_called)
    batch = dataset[0]
    assert isinstance(batch, VinOracleBatch)  # noqa: S101


def test_vin_offline_datamodule_supports_worker_batching(tmp_path: Path) -> None:
    """Exercise multi-worker batching against the immutable VIN store."""

    if not _supports_worker_tensor_sharing():
        pytest.skip("Host multiprocessing backend does not support worker tensor sharing.")

    store_cfg = _write_test_store(tmp_path)
    prior_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    dm_cfg = VinDataModuleConfig(
        source=VinOfflineSourceConfig(
            offline=VinOfflineDatasetConfig(store=store_cfg),
            train_split="train",
            val_split="val",
        ),
        batch_size=2,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        use_train_as_val=False,
    )
    try:
        datamodule = dm_cfg.setup_target()
        datamodule.setup(stage=Stage.TRAIN)

        train_batch = next(iter(datamodule.train_dataloader()))
        val_batch = next(iter(datamodule.val_dataloader()))
        assert isinstance(train_batch, VinOracleBatch)  # noqa: S101
        assert train_batch.rri.shape == (2, 4)  # noqa: S101
        assert torch.equal(train_batch.candidate_count, torch.tensor([2, 3], dtype=torch.int64))  # noqa: S101
        assert torch.equal(
            train_batch.candidate_valid_mask(),
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, False],
                ],
                dtype=torch.bool,
            ),
        )  # noqa: S101
        assert train_batch.scene_id == ["scene-a", "scene-b"]  # noqa: S101

        assert isinstance(val_batch, VinOracleBatch)  # noqa: S101
        assert val_batch.rri.shape == (1, 4)  # noqa: S101
        assert torch.equal(val_batch.candidate_count, torch.tensor([2], dtype=torch.int64))  # noqa: S101
        assert val_batch.scene_id == ["scene-c"]  # noqa: S101
    finally:
        torch.multiprocessing.set_sharing_strategy(prior_strategy)


def test_vin_offline_source_config_disables_diagnostic_blocks_for_vin_batches(tmp_path: Path) -> None:
    """The canonical offline source should expose the lean VIN-batch runtime path."""

    store_cfg = _write_test_store(tmp_path)
    dataset = VinOfflineSourceConfig(
        offline=VinOfflineDatasetConfig(store=store_cfg),
        train_split="train",
        val_split="val",
    ).setup_target(split=Stage.TRAIN)

    assert dataset.config.return_format == "vin_batch"  # noqa: S101
    assert dataset.config.load_candidates is False  # noqa: S101
    assert dataset.config.load_depths is False  # noqa: S101
    assert dataset.config.load_candidate_pcs is False  # noqa: S101
    assert dataset.config.load_counterfactuals is False  # noqa: S101

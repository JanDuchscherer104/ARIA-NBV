"""Focused round-trip tests for the immutable VIN offline dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aria_nbv.data_handling import (
    VinOfflineDatasetConfig,
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
    VinOfflineShardSpec,
    VinOfflineSourceConfig,
    VinOfflineStoreConfig,
    VinOracleBatch,
    VinSnippetView,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from aria_nbv.lightning.lit_datamodule import VinDataModuleConfig
from aria_nbv.rendering.candidate_depth_renderer import CandidateDepths
from aria_nbv.rri_metrics.types import RriResult
from aria_nbv.utils import Stage

PoseTW = pytest.importorskip("efm3d.aria.pose").PoseTW
PerspectiveCameras = pytest.importorskip(
    "pytorch3d.renderer.cameras",
).PerspectiveCameras


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
    shard_spec = VinOfflineShardSpec.from_dict(shard_spec.to_dict())
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
        version=1,
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
    store_cfg.sample_index_path.write_text(
        "".join(f"{record.to_json()}\n" for record in index_records),
        encoding="utf-8",
    )
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
    assert int(first.oracle.rri.shape[0]) == 2  # noqa: S101
    assert int(first.vin_snippet.lengths[0].item()) == 2  # noqa: S101

    sample_index_records = [
        json.loads(line)
        for line in store_cfg.sample_index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sample_index_records[0]["split"] == "train"  # noqa: S101
    assert sample_index_records[1]["split"] == "train"  # noqa: S101
    assert sample_index_records[2]["split"] == "val"  # noqa: S101

    batch_dataset = VinOfflineDatasetConfig(
        store=store_cfg,
        return_format="vin_batch",
        split="train",
    ).setup_target()
    batch = batch_dataset[0]
    assert isinstance(batch, VinOracleBatch)  # noqa: S101
    assert batch.scene_id == "scene-a"  # noqa: S101
    assert int(batch.rri.shape[0]) == 2  # noqa: S101


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
        assert train_batch.scene_id == ["scene-a", "scene-b"]  # noqa: S101

        assert isinstance(val_batch, VinOracleBatch)  # noqa: S101
        assert val_batch.scene_id == "scene-c"  # noqa: S101
    finally:
        torch.multiprocessing.set_sharing_strategy(prior_strategy)

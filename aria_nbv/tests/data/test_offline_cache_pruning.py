"""Tests for pruned offline cache loading in vin_batch mode.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
Delete this test with the legacy oracle-cache runtime.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

if TYPE_CHECKING:
    from pathlib import Path

import torch


def _write_dummy_metadata(cache_dir: Path) -> None:
    meta = {
        "version": 1,
        "created_at": "2024-01-01T00:00:00Z",
        "labeler_config": {},
        "labeler_signature": "dummy",
        "dataset_config": None,
        "backbone_config": None,
        "backbone_signature": None,
        "config_hash": "dummy",
        "include_backbone": True,
        "include_depths": True,
        "include_pointclouds": True,
        "num_samples": None,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_cache_dataset_prunes_candidate_pcs(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Skip candidate decoding when return_format is vin_batch."""
    if importlib.util.find_spec("power_spherical") is None:
        pytest.skip("Missing power_spherical dependency")
    offline_mod = importlib.import_module("aria_nbv.data_handling.oracle_cache")

    cache_dir = tmp_path / "cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True)
    _write_dummy_metadata(cache_dir)

    payload = {
        "scene_id": "scene",
        "snippet_id": "snippet",
        "candidates": {"dummy": 1},
        "depths": {"dummy": 2},
        "candidate_pcs": {"dummy": 3},
        "rri": {"dummy": 4},
    }
    sample_path = samples_dir / "sample.pt"
    torch.save(payload, sample_path)

    entry = {
        "key": "sample",
        "scene_id": "scene",
        "snippet_id": "snippet",
        "path": "samples/sample.pt",
    }
    (cache_dir / "index.jsonl").write_text(
        json.dumps(entry) + "\n",
        encoding="utf-8",
    )

    depths_stub = SimpleNamespace(
        poses="poses",
        reference_pose="ref_pose",
        p3d_cameras="cams",
    )
    rri_stub = SimpleNamespace(
        rri=torch.zeros(1),
        pm_dist_before=torch.zeros(1),
        pm_dist_after=torch.zeros(1),
        pm_acc_before=torch.zeros(1),
        pm_comp_before=torch.zeros(1),
        pm_acc_after=torch.zeros(1),
        pm_comp_after=torch.zeros(1),
    )

    def _decode_depths(
        _payload: dict[str, object],
        *,
        device: object,
    ) -> SimpleNamespace:
        _ = device
        return depths_stub

    def _decode_rri(
        _payload: dict[str, object],
        *,
        device: object,
    ) -> SimpleNamespace:
        _ = device
        return rri_stub

    def _decode_candidate_pcs(
        _payload: dict[str, object],
        *,
        device: object,
    ) -> NoReturn:
        _ = device
        raise AssertionError("decode_candidate_pcs should not be called")

    def _decode_candidates(_payload: dict[str, object]) -> NoReturn:
        raise AssertionError("decode_candidates should not be called")

    monkeypatch.setattr(offline_mod, "decode_depths", _decode_depths)
    monkeypatch.setattr(offline_mod, "decode_rri", _decode_rri)
    monkeypatch.setattr(offline_mod, "decode_candidate_pcs", _decode_candidate_pcs)
    monkeypatch.setattr(offline_mod, "decode_candidates", _decode_candidates)

    cache_cfg = offline_mod.OracleRriCacheDatasetConfig(
        cache=offline_mod.OracleRriCacheConfig(cache_dir=cache_dir),
        load_backbone=False,
        include_efm_snippet=False,
        return_format="vin_batch",
        load_candidates=False,
        load_candidate_pcs=False,
        load_depths=True,
    )
    cache_ds = cache_cfg.setup_target()
    batch = cache_ds[0]

    assert getattr(batch, "scene_id", None) == "scene"  # noqa: S101
    assert getattr(batch, "snippet_id", None) == "snippet"  # noqa: S101
    assert batch.candidate_poses_world_cam == "poses"  # noqa: S101

"""Tests for VIN snippet cache integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)

import aria_nbv.data.offline_cache as offline_cache
import aria_nbv.data.vin_snippet_cache as vin_snippet_cache
from aria_nbv.configs import PathConfig
from aria_nbv.data import VinSnippetView
from aria_nbv.data.offline_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
)
from aria_nbv.data.offline_cache_serialization import encode_rri
from aria_nbv.data.offline_cache_store import _write_metadata
from aria_nbv.data.offline_cache_types import OracleRriCacheMetadata
from aria_nbv.data.vin_snippet_cache import (
    VinSnippetCacheConfig,
    VinSnippetCacheWriterConfig,
)
from aria_nbv.data.vin_snippet_utils import vin_snippet_cache_config_hash
from aria_nbv.rendering.candidate_depth_renderer import CandidateDepths
from aria_nbv.rri_metrics.types import RriResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_pose_batch(num: int) -> PoseTW:
    r = torch.eye(3).expand(num, 3, 3).clone()
    t = torch.zeros((num, 3))
    return PoseTW.from_Rt(r, t)


def _make_stub_depths(num_candidates: int) -> CandidateDepths:
    h, w = 4, 4
    depths = torch.ones((num_candidates, h, w))
    depths_valid = torch.ones_like(depths, dtype=torch.bool)
    poses = _make_pose_batch(num_candidates)
    ref_pose = _make_pose_batch(1).squeeze(0)

    r = torch.eye(3).expand(num_candidates, 3, 3).clone()
    t = torch.zeros((num_candidates, 3))
    focal = torch.full((num_candidates, 2), 50.0)
    principal = torch.full((num_candidates, 2), 2.0)
    image_size = torch.full((num_candidates, 2), 4.0)
    p3d = PerspectiveCameras(
        R=r,
        T=t,
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


def _decode_depths_stub(
    payload: dict[str, object],
    *,
    device: torch.device,
) -> CandidateDepths:
    _ = device
    return _make_stub_depths(int(payload.get("num_candidates", 2)))


def _write_vin_snippet_metadata(  # noqa: PLR0913
    path: Path,
    *,
    dataset_config: dict[str, object] | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    semidense_max_points: int | None,
    num_samples: int | None,
) -> None:
    config_hash = vin_snippet_cache_config_hash(
        dataset_config=dataset_config,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        semidense_max_points=semidense_max_points,
        pad_points=vin_snippet_cache.VIN_SNIPPET_PAD_POINTS,
    )
    payload = {
        "version": 1,
        "created_at": "2026-01-03",
        "source_cache_dir": None,
        "source_cache_hash": None,
        "dataset_config": dataset_config,
        "include_inv_dist_std": include_inv_dist_std,
        "include_obs_count": include_obs_count,
        "semidense_max_points": semidense_max_points,
        "pad_points": vin_snippet_cache.VIN_SNIPPET_PAD_POINTS,
        "config_hash": config_hash,
        "num_samples": num_samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_read_vin_snippet_cache_metadata_accepts_cache_dir(tmp_path: Path) -> None:
    """Allow callers to pass the VIN snippet cache directory.

    This avoids requiring callers to pass `metadata.json` explicitly.
    """
    cache_dir = tmp_path / "vin_snippet_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    expected_num_samples = 3

    _write_vin_snippet_metadata(
        meta_path,
        dataset_config=None,
        include_inv_dist_std=True,
        include_obs_count=False,
        semidense_max_points=None,
        num_samples=expected_num_samples,
    )

    meta = vin_snippet_cache.read_vin_snippet_cache_metadata(cache_dir)
    assert meta.version == 1  # noqa: S101
    assert meta.num_samples == expected_num_samples  # noqa: S101


def test_oracle_cache_uses_vin_snippet_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer VIN snippet cache when available."""
    cache_dir = tmp_path / "oracle_cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    meta = OracleRriCacheMetadata(
        version=1,
        created_at="2026-01-03",
        labeler_config={},
        labeler_signature="stub",
        dataset_config={},
        backbone_config=None,
        backbone_signature=None,
        config_hash="test",
        include_backbone=False,
        include_depths=True,
        include_pointclouds=False,
        num_samples=1,
    )
    _write_metadata(cache_dir / "metadata.json", meta)

    rri = RriResult(
        rri=torch.tensor([0.1, 0.2]),
        pm_dist_before=torch.full((2,), 0.5),
        pm_dist_after=torch.full((2,), 0.4),
        pm_acc_before=torch.full((2,), 0.3),
        pm_comp_before=torch.full((2,), 0.2),
        pm_acc_after=torch.full((2,), 0.25),
        pm_comp_after=torch.full((2,), 0.15),
    )
    payload = {
        "scene_id": "scene_0",
        "snippet_id": "000001",
        "rri": encode_rri(rri),
        "depths": {"num_candidates": 2},
    }
    sample_path = samples_dir / "sample_0.pt"
    torch.save(payload, sample_path)
    (cache_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "key": "sample_0",
                "scene_id": "scene_0",
                "snippet_id": "000001",
                "path": "samples/sample_0.pt",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    vin_cache_dir = tmp_path / "vin_cache"
    vin_samples_dir = vin_cache_dir / "samples"
    vin_samples_dir.mkdir(parents=True, exist_ok=True)
    points_world = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.5]],
        dtype=torch.float32,
    )
    t_world_rig = _make_pose_batch(2).tensor()
    vin_payload = {
        "scene_id": "scene_0",
        "snippet_id": "000001",
        "points_world": points_world,
        "t_world_rig": t_world_rig,
    }
    torch.save(vin_payload, vin_samples_dir / "vin_0.pt")
    (vin_cache_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "key": "vin_0",
                "scene_id": "scene_0",
                "snippet_id": "000001",
                "path": "samples/vin_0.pt",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    _write_vin_snippet_metadata(
        vin_cache_dir / "metadata.json",
        dataset_config={},
        include_inv_dist_std=True,
        include_obs_count=False,
        semidense_max_points=None,
        num_samples=1,
    )

    monkeypatch.setattr(
        "aria_nbv.data_handling.oracle_cache.decode_depths",
        _decode_depths_stub,
    )

    def _raise_load(self: object, *, scene_id: str, snippet_id: str) -> None:  # noqa: ARG001
        raise RuntimeError(
            "EFM snippet load should not be called when vin_snippet_cache is set.",
        )

    monkeypatch.setattr(offline_cache.EfmSnippetLoader, "load", _raise_load)

    cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig()),
        include_efm_snippet=True,
        load_backbone=False,
        load_depths=True,
        load_candidates=False,
        load_candidate_pcs=False,
        return_format="vin_batch",
        vin_snippet_cache=VinSnippetCacheConfig(
            cache_dir=vin_cache_dir,
            paths=PathConfig(),
        ),
    )
    dataset = cfg.setup_target()
    batch = dataset[0]

    assert isinstance(batch.efm_snippet_view, VinSnippetView)  # noqa: S101
    assert torch.allclose(  # noqa: S101
        batch.efm_snippet_view.points_world.cpu(),
        points_world,
    )
    assert batch.efm_snippet_view.t_world_rig.tensor().shape == t_world_rig.shape  # noqa: S101


def test_oracle_cache_filters_to_vin_snippet_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restrict dataset entries to those present in VIN snippet cache."""
    cache_dir = tmp_path / "oracle_cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    meta = OracleRriCacheMetadata(
        version=1,
        created_at="2026-01-03",
        labeler_config={},
        labeler_signature="stub",
        dataset_config={},
        backbone_config=None,
        backbone_signature=None,
        config_hash="test",
        include_backbone=False,
        include_depths=True,
        include_pointclouds=False,
        num_samples=2,
    )
    _write_metadata(cache_dir / "metadata.json", meta)

    rri = RriResult(
        rri=torch.tensor([0.1, 0.2]),
        pm_dist_before=torch.full((2,), 0.5),
        pm_dist_after=torch.full((2,), 0.4),
        pm_acc_before=torch.full((2,), 0.3),
        pm_comp_before=torch.full((2,), 0.2),
        pm_acc_after=torch.full((2,), 0.25),
        pm_comp_after=torch.full((2,), 0.15),
    )
    for idx, snippet_id in enumerate(["000001", "000002"]):
        payload = {
            "scene_id": "scene_0",
            "snippet_id": snippet_id,
            "rri": encode_rri(rri),
            "depths": {"num_candidates": 2},
        }
        sample_path = samples_dir / f"sample_{idx}.pt"
        torch.save(payload, sample_path)

    (cache_dir / "index.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "key": f"sample_{idx}",
                    "scene_id": "scene_0",
                    "snippet_id": snippet_id,
                    "path": f"samples/sample_{idx}.pt",
                },
            )
            for idx, snippet_id in enumerate(["000001", "000002"])
        )
        + "\n",
        encoding="utf-8",
    )

    vin_cache_dir = tmp_path / "vin_cache"
    vin_samples_dir = vin_cache_dir / "samples"
    vin_samples_dir.mkdir(parents=True, exist_ok=True)
    points_world = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    t_world_rig = _make_pose_batch(1).tensor()
    vin_payload = {
        "scene_id": "scene_0",
        "snippet_id": "000001",
        "points_world": points_world,
        "t_world_rig": t_world_rig,
    }
    torch.save(vin_payload, vin_samples_dir / "vin_0.pt")
    (vin_cache_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "key": "vin_0",
                "scene_id": "scene_0",
                "snippet_id": "000001",
                "path": "samples/vin_0.pt",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    _write_vin_snippet_metadata(
        vin_cache_dir / "metadata.json",
        dataset_config={},
        include_inv_dist_std=True,
        include_obs_count=False,
        semidense_max_points=None,
        num_samples=1,
    )

    monkeypatch.setattr(
        "aria_nbv.data_handling.oracle_cache.decode_depths",
        _decode_depths_stub,
    )

    cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig()),
        include_efm_snippet=True,
        load_backbone=False,
        load_depths=True,
        load_candidates=False,
        load_candidate_pcs=False,
        return_format="vin_batch",
        vin_snippet_cache_mode="required",
        vin_snippet_cache_allow_subset=True,
        vin_snippet_cache=VinSnippetCacheConfig(
            cache_dir=vin_cache_dir,
            paths=PathConfig(),
        ),
    )
    dataset = cfg.setup_target()

    assert len(dataset) == 1  # noqa: S101
    batch = dataset[0]
    assert batch.scene_id == "scene_0"  # noqa: S101
    assert batch.snippet_id == "000001"  # noqa: S101


def test_vin_snippet_cache_writer_uses_dataloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure DataLoader-based VIN cache building writes samples."""
    cache_dir = tmp_path / "oracle_cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    meta = OracleRriCacheMetadata(
        version=1,
        created_at="2026-01-05",
        labeler_config={},
        labeler_signature="stub",
        dataset_config={},
        backbone_config=None,
        backbone_signature=None,
        config_hash="test",
        include_backbone=False,
        include_depths=False,
        include_pointclouds=False,
        num_samples=1,
    )
    _write_metadata(cache_dir / "metadata.json", meta)
    (cache_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "key": "sample_0",
                "scene_id": "scene_0",
                "snippet_id": "000001",
                "path": "samples/sample_0.pt",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    vin_cache_dir = tmp_path / "vin_cache"
    vin_samples_dir = vin_cache_dir / "samples"
    vin_samples_dir.mkdir(parents=True, exist_ok=True)
    points_world = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    t_world_rig = _make_pose_batch(1).tensor()

    def _stub_build_payload(self: object, entry: object) -> dict[str, object]:  # noqa: ARG001
        return {
            "scene_id": "scene_0",
            "snippet_id": "000001",
            "points_world": points_world,
            "t_world_rig": t_world_rig,
        }

    monkeypatch.setattr(
        vin_snippet_cache.VinSnippetCacheBuildDataset,
        "_build_payload",
        _stub_build_payload,
    )

    cfg = VinSnippetCacheWriterConfig(
        paths=PathConfig(),
        cache=VinSnippetCacheConfig(cache_dir=vin_cache_dir, paths=PathConfig()),
        source_cache=OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig()),
        use_dataloader=True,
        num_workers=0,
        overwrite=True,
        resume=True,
    )
    entries = cfg.setup_target().run()

    assert len(entries) == 1  # noqa: S101
    saved = torch.load(vin_cache_dir / entries[0].path, weights_only=False)
    assert saved["scene_id"] == "scene_0"  # noqa: S101


def test_vin_snippet_cache_writer_skips_missing_snippets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing snippets should be skipped when configured."""
    cache_dir = tmp_path / "oracle_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.jsonl").write_text(
        json.dumps(
            {
                "key": "sample_0",
                "scene_id": "scene_0",
                "snippet_id": "000001",
                "path": "samples/sample_0.pt",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    meta = OracleRriCacheMetadata(
        version=1,
        created_at="2026-01-05",
        labeler_config={},
        labeler_signature="stub",
        dataset_config={},
        backbone_config=None,
        backbone_signature=None,
        config_hash="test",
        include_backbone=False,
        include_depths=False,
        include_pointclouds=False,
        num_samples=1,
    )
    _write_metadata(cache_dir / "metadata.json", meta)

    def _raise_missing(self: object, entry: object) -> dict[str, object]:  # noqa: ARG001
        raise FileNotFoundError("missing snippet")

    monkeypatch.setattr(
        vin_snippet_cache.VinSnippetCacheBuildDataset,
        "_build_payload",
        _raise_missing,
    )

    vin_cache_dir = tmp_path / "vin_cache"
    vin_cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = VinSnippetCacheWriterConfig(
        paths=PathConfig(),
        cache=VinSnippetCacheConfig(cache_dir=vin_cache_dir, paths=PathConfig()),
        source_cache=OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig()),
        use_dataloader=True,
        num_workers=0,
        overwrite=True,
        resume=True,
        skip_missing_snippets=True,
        num_failures_allowed=0,
    )
    entries = cfg.setup_target().run()
    assert entries == []  # noqa: S101

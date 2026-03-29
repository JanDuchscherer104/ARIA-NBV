"""Tests for the v2 cache/data pipeline under ``aria_nbv.data_handling``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("efm3d")
pytest.importorskip("pytorch3d")

import torch
from efm3d.aria.aria_constants import (
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_INV_DIST_STD,
    ARIA_POINTS_TIME_NS,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
    ARIA_POSE_T_WORLD_RIG,
    ARIA_POSE_TIME_NS,
)
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

import aria_nbv.data_handling.efm_snippet_loader as efm_loader_mod
import aria_nbv.data_handling.oracle_cache as oracle_cache_mod
from aria_nbv.data_handling.cache_index import read_index
from aria_nbv.data_handling.efm_views import EfmSnippetView, VinSnippetView
from aria_nbv.data_handling.offline_cache_serialization import encode_rri
from aria_nbv.data_handling.offline_cache_store import _write_metadata
from aria_nbv.data_handling.oracle_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
    OracleRriCacheMetadata,
    rebuild_oracle_cache_index,
    repair_oracle_cache_indices,
)
from aria_nbv.data_handling.vin_adapter import build_vin_snippet_view
from aria_nbv.data_handling.vin_cache import (
    VinSnippetCacheConfig,
    VinSnippetCacheEntry,
    VinSnippetCacheWriterConfig,
    read_vin_snippet_cache_metadata,
    rebuild_vin_snippet_cache_index,
)
from aria_nbv.data_handling.vin_oracle_datasets import VinOracleCacheDatasetConfig
from aria_nbv.data_handling.vin_oracle_types import VinOracleBatch
from aria_nbv.lightning.lit_datamodule import VinDataModuleConfig
from aria_nbv.rendering.candidate_depth_renderer import CandidateDepths
from aria_nbv.rri_metrics.types import RriResult
from aria_nbv.utils import Verbosity


def _make_pose_batch(num: int, *, offset: float = 0.0) -> PoseTW:
    rotation = torch.eye(3, dtype=torch.float32).expand(num, 3, 3).clone()
    translation = torch.zeros((num, 3), dtype=torch.float32)
    translation[:, 0] = offset
    translation[:, 1] = torch.arange(num, dtype=torch.float32)
    return PoseTW.from_Rt(rotation, translation)


def _make_stub_depths(num_candidates: int) -> CandidateDepths:
    depths = torch.ones((num_candidates, 4, 4), dtype=torch.float32)
    depths_valid = torch.ones_like(depths, dtype=torch.bool)
    poses = _make_pose_batch(num_candidates)
    ref_pose = _make_pose_batch(1).squeeze(0)

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


def _decode_depths_stub(
    payload: dict[str, object],
    *,
    device: torch.device,
) -> CandidateDepths:
    _ = device
    return _make_stub_depths(int(payload.get("num_candidates", 2)))


def _make_stub_rri(num_candidates: int = 2) -> RriResult:
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


def _make_efm_snippet(
    *,
    scene_id: str,
    snippet_id: str,
    offset: float = 0.0,
) -> EfmSnippetView:
    points_world = torch.tensor(
        [
            [
                [offset + 0.0, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [float("nan"), float("nan"), float("nan")],
            ],
            [
                [offset + 2.0, 0.0, 0.0],
                [float("nan"), float("nan"), float("nan")],
                [float("nan"), float("nan"), float("nan")],
            ],
        ],
        dtype=torch.float32,
    )
    inv_dist_std = torch.tensor(
        [
            [0.1, 0.2, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    efm = {
        ARIA_POSE_T_WORLD_RIG: _make_pose_batch(2, offset=offset),
        ARIA_POSE_TIME_NS: torch.tensor([10, 20], dtype=torch.int64),
        "pose/gravity_in_world": torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32),
        ARIA_POINTS_WORLD: points_world,
        ARIA_POINTS_DIST_STD: torch.ones((2, 3), dtype=torch.float32),
        ARIA_POINTS_INV_DIST_STD: inv_dist_std,
        ARIA_POINTS_TIME_NS: torch.tensor([10, 20], dtype=torch.int64),
        ARIA_POINTS_VOL_MIN: torch.tensor([offset, 0.0, 0.0], dtype=torch.float32),
        ARIA_POINTS_VOL_MAX: torch.tensor([offset + 2.0, 1.0, 1.0], dtype=torch.float32),
        "points/lengths": torch.tensor([2, 1], dtype=torch.int64),
    }
    return EfmSnippetView(efm=efm, scene_id=scene_id, snippet_id=snippet_id)


def _write_oracle_metadata(
    cache_dir: Path,
    *,
    num_samples: int | None,
    dataset_config: dict[str, Any] | None = None,
) -> None:
    meta = OracleRriCacheMetadata(
        version=1,
        created_at="2026-03-29T00:00:00Z",
        labeler_config={},
        labeler_signature="stub",
        dataset_config=dataset_config,
        backbone_config=None,
        backbone_signature=None,
        config_hash="oracle-cache-test",
        include_backbone=False,
        include_depths=True,
        include_pointclouds=False,
        num_samples=num_samples,
    )
    _write_metadata(cache_dir / "metadata.json", meta)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _oracle_sample_payload(scene_id: str, snippet_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "snippet_id": snippet_id,
        "rri": encode_rri(_make_stub_rri()),
        "depths": {"num_candidates": 2},
    }


def _write_oracle_cache(
    cache_dir: Path,
    *,
    pairs: list[tuple[str, str]],
    write_payloads: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    _write_oracle_metadata(cache_dir, num_samples=len(pairs), dataset_config={})

    rows: list[dict[str, str]] = []
    for idx, (scene_id, snippet_id) in enumerate(pairs):
        rel_path = f"samples/sample_{idx}.pt"
        if write_payloads:
            torch.save(
                _oracle_sample_payload(scene_id, snippet_id),
                cache_dir / rel_path,
            )
        rows.append(
            {
                "key": f"sample_{idx}",
                "scene_id": scene_id,
                "snippet_id": snippet_id,
                "path": rel_path,
            },
        )
    _write_jsonl(cache_dir / "index.jsonl", rows)


def _as_vin_snippet(batch: VinOracleBatch) -> VinSnippetView:
    view = batch.efm_snippet_view
    if not isinstance(view, VinSnippetView):
        raise TypeError(f"Expected VinSnippetView, got {type(view)}.")
    return view


def _first_train_batch(config: VinDataModuleConfig) -> VinOracleBatch:
    datamodule = config.setup_target()
    return next(iter(datamodule.train_dataloader()))


def _make_datamodule_cfg(
    *,
    oracle_cache: OracleRriCacheConfig,
    vin_cache: VinSnippetCacheConfig,
    batch_size: int | None,
    limit: int,
    vin_mode: str,
) -> VinDataModuleConfig:
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=oracle_cache,
        split="all",
        limit=limit,
        include_efm_snippet=True,
        include_gt_mesh=False,
        load_backbone=False,
        load_candidates=False,
        load_depths=True,
        load_candidate_pcs=False,
        vin_snippet_cache=vin_cache,
        vin_snippet_cache_mode=vin_mode,
    )
    return VinDataModuleConfig(
        source=VinOracleCacheDatasetConfig(
            cache=cache_cfg,
            train_split="all",
            val_split="all",
        ),
        use_train_as_val=True,
        shuffle=False,
        shuffle_candidates=False,
        num_workers=0,
        batch_size=batch_size,
        persistent_workers=False,
        verbosity=Verbosity.QUIET,
    )


def test_build_vin_snippet_view_collapses_and_pads_once() -> None:
    """Collapse padded per-frame semidense points once, then pad once."""
    efm_snippet = _make_efm_snippet(scene_id="scene_0", snippet_id="000001")

    vin_snippet = build_vin_snippet_view(
        efm_snippet,
        device=torch.device("cpu"),
        max_points=None,
        include_inv_dist_std=True,
        include_obs_count=False,
        pad_points=5,
    )

    assert vin_snippet.points_world.shape == (5, 4)  # noqa: S101
    assert vin_snippet.lengths.tolist() == [3]  # noqa: S101
    torch.testing.assert_close(
        vin_snippet.points_world[:3],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.1],
                [1.0, 0.0, 0.0, 0.2],
                [2.0, 0.0, 0.0, 0.3],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.isnan(vin_snippet.points_world[3:]).all()  # noqa: S101
    assert vin_snippet.t_world_rig.tensor().shape == (2, 12)  # noqa: S101


def test_oracle_cache_reader_requires_fresh_split_indices(tmp_path: Path) -> None:
    """Split readers should fail clearly on missing or stale split files."""
    cache_dir = tmp_path / "oracle_cache"
    pairs = [("scene_0", "000001"), ("scene_0", "000002")]
    _write_oracle_cache(cache_dir, pairs=pairs, write_payloads=False)
    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)

    with pytest.raises(FileNotFoundError, match="repair_oracle_cache_indices"):
        OracleRriCacheDatasetConfig(
            cache=cache_cfg,
            split="train",
            include_efm_snippet=False,
            load_backbone=False,
            load_candidates=False,
            load_depths=True,
            load_candidate_pcs=False,
        ).setup_target()

    repair_oracle_cache_indices(cache=cache_cfg, train_val_split=0.5)

    with (cache_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": "sample_2",
                    "scene_id": "scene_0",
                    "snippet_id": "000003",
                    "path": "samples/sample_2.pt",
                },
            )
            + "\n",
        )

    with pytest.raises(ValueError, match="stale relative to `index.jsonl`"):
        OracleRriCacheDatasetConfig(
            cache=cache_cfg,
            split="train",
            include_efm_snippet=False,
            load_backbone=False,
            load_candidates=False,
            load_depths=True,
            load_candidate_pcs=False,
        ).setup_target()


def test_rebuild_oracle_cache_index_writes_base_and_split_indices(tmp_path: Path) -> None:
    """Oracle rebuild should rewrite base/train/val indices and metadata counts."""
    cache_dir = tmp_path / "oracle_cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    _write_oracle_metadata(cache_dir, num_samples=None, dataset_config={})

    for idx in range(4):
        sample_path = samples_dir / f"ASE_NBV_SNIPPET_scene_00000{idx}_hash.pt"
        sample_path.write_text("x", encoding="utf-8")

    count = rebuild_oracle_cache_index(
        cache_dir=cache_dir,
        train_val_split=0.25,
        rng_seed=123,
    )

    assert count == 4  # noqa: S101
    assert len(_read_jsonl(cache_dir / "index.jsonl")) == 4  # noqa: S101
    assert len(_read_jsonl(cache_dir / "train_index.jsonl")) == 3  # noqa: S101
    assert len(_read_jsonl(cache_dir / "val_index.jsonl")) == 1  # noqa: S101

    meta_payload = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta_payload["num_samples"] == 4  # noqa: S101


def test_vin_snippet_cache_writer_rewrites_index_for_resume_and_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIN cache writer should append on resume and replace cleanly on overwrite."""
    oracle_cache_dir = tmp_path / "oracle_cache"
    pairs = [("scene_0", "000001"), ("scene_0", "000002")]
    _write_oracle_cache(oracle_cache_dir, pairs=pairs, write_payloads=False)

    snippets = {
        pair: _make_efm_snippet(scene_id=pair[0], snippet_id=pair[1], offset=float(idx) * 10.0)
        for idx, pair in enumerate(pairs)
    }

    def _load(self: object, *, scene_id: str, snippet_id: str) -> EfmSnippetView:
        return snippets[(scene_id, snippet_id)]

    monkeypatch.setattr(efm_loader_mod.EfmSnippetLoader, "load", _load)

    vin_cache_dir = tmp_path / "vin_cache"
    cache_cfg = VinSnippetCacheConfig(cache_dir=vin_cache_dir, pad_points=6)
    writer_kwargs = {
        "cache": cache_cfg,
        "source_cache": OracleRriCacheConfig(cache_dir=oracle_cache_dir),
        "map_location": "cpu",
        "semidense_max_points": None,
        "verbosity": Verbosity.QUIET,
    }

    first_entries = (
        VinSnippetCacheWriterConfig(
            max_samples=1,
            overwrite=False,
            resume=True,
            **writer_kwargs,
        )
        .setup_target()
        .run()
    )
    assert len(first_entries) == 1  # noqa: S101
    assert len(read_index(cache_cfg.index_path, entry_type=VinSnippetCacheEntry)) == 1  # noqa: S101
    assert read_vin_snippet_cache_metadata(cache_cfg.cache_dir).num_samples == 1  # noqa: S101

    second_entries = (
        VinSnippetCacheWriterConfig(
            max_samples=2,
            overwrite=False,
            resume=True,
            **writer_kwargs,
        )
        .setup_target()
        .run()
    )
    assert len(second_entries) == 1  # noqa: S101
    assert len(read_index(cache_cfg.index_path, entry_type=VinSnippetCacheEntry)) == 2  # noqa: S101
    assert read_vin_snippet_cache_metadata(cache_cfg.cache_dir).num_samples == 2  # noqa: S101

    overwrite_entries = (
        VinSnippetCacheWriterConfig(
            max_samples=1,
            overwrite=True,
            resume=False,
            **writer_kwargs,
        )
        .setup_target()
        .run()
    )
    assert len(overwrite_entries) == 1  # noqa: S101
    assert len(read_index(cache_cfg.index_path, entry_type=VinSnippetCacheEntry)) == 1  # noqa: S101
    assert len(list(cache_cfg.samples_dir.glob("*.pt"))) == 1  # noqa: S101
    assert read_vin_snippet_cache_metadata(cache_cfg.cache_dir).num_samples == 1  # noqa: S101


def test_rebuild_vin_snippet_cache_index_scans_payloads_and_updates_metadata(
    tmp_path: Path,
) -> None:
    """VIN index rebuild should recover entries from payload files."""
    cache_cfg = VinSnippetCacheConfig(cache_dir=tmp_path / "vin_cache", pad_points=7)
    cache_cfg.samples_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_cfg.metadata_path
    meta_path.write_text(
        json.dumps(
            {
                "version": 2,
                "created_at": "2026-03-29T00:00:00Z",
                "source_cache_dir": None,
                "source_cache_hash": None,
                "dataset_config": {},
                "include_inv_dist_std": True,
                "include_obs_count": False,
                "semidense_max_points": None,
                "pad_points": 7,
                "config_hash": "stub",
                "num_samples": None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for idx in range(2):
        torch.save(
            {
                "scene_id": "scene_0",
                "snippet_id": f"00000{idx}",
                "points_world": torch.full((7, 4), float(idx), dtype=torch.float32),
                "points_length": torch.tensor([3], dtype=torch.int64),
                "t_world_rig": _make_pose_batch(2, offset=float(idx)).tensor(),
            },
            cache_cfg.samples_dir / f"vin_{idx}.pt",
        )

    count = rebuild_vin_snippet_cache_index(cache_dir=cache_cfg.cache_dir, pad_points=7)

    assert count == 2  # noqa: S101
    assert len(read_index(cache_cfg.index_path, entry_type=VinSnippetCacheEntry)) == 2  # noqa: S101
    meta = read_vin_snippet_cache_metadata(cache_cfg.cache_dir)
    assert meta.num_samples == 2  # noqa: S101
    assert meta.pad_points == 7  # noqa: S101


def test_oracle_cache_live_fallback_matches_vin_cache_direct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live EFM fallback and VIN cache should produce identical VIN snippets."""
    oracle_cache_dir = tmp_path / "oracle_cache"
    pairs = [("scene_0", "000001")]
    _write_oracle_cache(oracle_cache_dir, pairs=pairs, write_payloads=True)

    snippets = {
        ("scene_0", "000001"): _make_efm_snippet(
            scene_id="scene_0",
            snippet_id="000001",
            offset=0.0,
        ),
    }

    def _load(self: object, *, scene_id: str, snippet_id: str) -> EfmSnippetView:
        return snippets[(scene_id, snippet_id)]

    monkeypatch.setattr(efm_loader_mod.EfmSnippetLoader, "load", _load)
    monkeypatch.setattr(oracle_cache_mod, "decode_depths", _decode_depths_stub)

    oracle_cache_cfg = OracleRriCacheConfig(cache_dir=oracle_cache_dir)
    vin_cache_cfg = VinSnippetCacheConfig(cache_dir=tmp_path / "vin_cache", pad_points=6)
    VinSnippetCacheWriterConfig(
        cache=vin_cache_cfg,
        source_cache=oracle_cache_cfg,
        overwrite=True,
        resume=False,
        map_location="cpu",
        verbosity=Verbosity.QUIET,
    ).setup_target().run()

    live_dataset = OracleRriCacheDatasetConfig(
        cache=oracle_cache_cfg,
        include_efm_snippet=True,
        load_backbone=False,
        load_candidates=False,
        load_depths=True,
        load_candidate_pcs=False,
        return_format="vin_batch",
        vin_snippet_cache=vin_cache_cfg,
        vin_snippet_cache_mode="disabled",
    ).setup_target()
    live_batch = live_dataset[0]
    live_view = _as_vin_snippet(live_batch)

    def _raise_load(self: object, *, scene_id: str, snippet_id: str) -> EfmSnippetView:  # noqa: ARG001
        raise RuntimeError("live EFM loader should not be used when VIN cache is required")

    monkeypatch.setattr(efm_loader_mod.EfmSnippetLoader, "load", _raise_load)

    cached_dataset = OracleRriCacheDatasetConfig(
        cache=oracle_cache_cfg,
        include_efm_snippet=True,
        load_backbone=False,
        load_candidates=False,
        load_depths=True,
        load_candidate_pcs=False,
        return_format="vin_batch",
        vin_snippet_cache=vin_cache_cfg,
        vin_snippet_cache_mode="required",
    ).setup_target()
    cached_batch = cached_dataset[0]
    cached_view = _as_vin_snippet(cached_batch)

    assert live_batch.scene_id == cached_batch.scene_id  # noqa: S101
    assert live_batch.snippet_id == cached_batch.snippet_id  # noqa: S101
    torch.testing.assert_close(live_view.points_world, cached_view.points_world, equal_nan=True)
    torch.testing.assert_close(live_view.lengths, cached_view.lengths)
    torch.testing.assert_close(
        live_view.t_world_rig.tensor(),
        cached_view.t_world_rig.tensor(),
    )


@pytest.mark.parametrize(
    ("batch_size", "limit"),
    [
        (None, 1),
        (2, 2),
    ],
)
def test_v2_datamodule_matches_live_and_cached_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int | None,
    limit: int,
) -> None:
    """Datamodule batches should match between live fallback and VIN cache."""
    oracle_cache_dir = tmp_path / "oracle_cache"
    pairs = [
        ("scene_0", "000001"),
        ("scene_0", "000002"),
    ]
    _write_oracle_cache(oracle_cache_dir, pairs=pairs, write_payloads=True)

    snippets = {
        pair: _make_efm_snippet(scene_id=pair[0], snippet_id=pair[1], offset=float(idx) * 10.0)
        for idx, pair in enumerate(pairs)
    }

    def _load(self: object, *, scene_id: str, snippet_id: str) -> EfmSnippetView:
        return snippets[(scene_id, snippet_id)]

    monkeypatch.setattr(efm_loader_mod.EfmSnippetLoader, "load", _load)
    monkeypatch.setattr(oracle_cache_mod, "decode_depths", _decode_depths_stub)

    oracle_cache_cfg = OracleRriCacheConfig(cache_dir=oracle_cache_dir)
    vin_cache_cfg = VinSnippetCacheConfig(cache_dir=tmp_path / "vin_cache", pad_points=6)
    VinSnippetCacheWriterConfig(
        cache=vin_cache_cfg,
        source_cache=oracle_cache_cfg,
        overwrite=True,
        resume=False,
        map_location="cpu",
        max_samples=limit,
        verbosity=Verbosity.QUIET,
    ).setup_target().run()

    live_cfg = _make_datamodule_cfg(
        oracle_cache=oracle_cache_cfg,
        vin_cache=vin_cache_cfg,
        batch_size=batch_size,
        limit=limit,
        vin_mode="disabled",
    )
    live_batch = _first_train_batch(live_cfg)
    live_view = _as_vin_snippet(live_batch)

    def _raise_load(self: object, *, scene_id: str, snippet_id: str) -> EfmSnippetView:  # noqa: ARG001
        raise RuntimeError("VIN cache path should not fall back to live EFM loading")

    monkeypatch.setattr(efm_loader_mod.EfmSnippetLoader, "load", _raise_load)

    cached_cfg = _make_datamodule_cfg(
        oracle_cache=oracle_cache_cfg,
        vin_cache=vin_cache_cfg,
        batch_size=batch_size,
        limit=limit,
        vin_mode="required",
    )
    cached_batch = _first_train_batch(cached_cfg)
    cached_view = _as_vin_snippet(cached_batch)

    assert live_batch.scene_id == cached_batch.scene_id  # noqa: S101
    assert live_batch.snippet_id == cached_batch.snippet_id  # noqa: S101
    torch.testing.assert_close(live_view.points_world, cached_view.points_world, equal_nan=True)
    torch.testing.assert_close(live_view.lengths, cached_view.lengths)
    torch.testing.assert_close(
        live_view.t_world_rig.tensor(),
        cached_view.t_world_rig.tensor(),
    )

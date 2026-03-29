"""Integration tests comparing OfflineCache vs VinSnippetCache via VinDataModule.

We exercise both code paths *through VinDataModule*:
- Offline oracle cache + on-demand EFM snippet load → VinSnippetView
  (built at read time)
- Offline oracle cache + VinSnippetCache → VinSnippetView
  (loaded from precomputed cache)

We validate equivalence for the VIN v2 snippet inputs: collapsed semidense
points (XYZ + inv_dist_std) and historical trajectory poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from aria_nbv.configs import PathConfig
from aria_nbv.data_handling import (
    VIN_SNIPPET_PAD_POINTS,
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
    VinOracleBatch,
    VinOracleCacheDatasetConfig,
    VinSnippetCacheConfig,
    VinSnippetCacheWriterConfig,
    VinSnippetView,
)
from aria_nbv.lightning.lit_datamodule import VinDataModuleConfig
from aria_nbv.utils import Verbosity

if TYPE_CHECKING:
    from pathlib import Path


_BATCHED_POINTS_NDIM = 3
_VIN_POINTS_DIM = 4


def _skip_if_missing_oracle_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        pytest.skip("Missing offline oracle cache directory.")
    if not (cache_dir / "index.jsonl").exists():
        pytest.skip("Missing offline cache index.jsonl.")
    if not (cache_dir / "metadata.json").exists():
        pytest.skip("Missing offline cache metadata.json.")
    if not (cache_dir / "samples").exists():
        pytest.skip("Missing offline cache samples directory.")


def _first_train_batch(config: VinDataModuleConfig) -> VinOracleBatch:
    datamodule = config.setup_target()
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    if not isinstance(batch, VinOracleBatch):
        msg = f"Expected VinOracleBatch from train_dataloader, got {type(batch)}."
        raise TypeError(msg)
    return batch


def _as_vin_snippet(batch: VinOracleBatch) -> VinSnippetView:
    view = batch.efm_snippet_view
    if not isinstance(view, VinSnippetView):
        msg = f"Expected VinSnippetView in batch.efm_snippet_view, got {type(view)}."
        raise TypeError(msg)
    return view


def _make_offline_datamodule_cfg(
    *,
    cache_dir: Path,
    vin_snippet_cache_dir: Path | None,
    split: str,
    limit: int,
    batch_size: int | None,
) -> VinDataModuleConfig:
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=cache_dir),
        split="all",
        limit=limit,
        include_efm_snippet=True,
        include_gt_mesh=False,
        load_backbone=True,
        load_candidates=False,
        load_depths=True,
        load_candidate_pcs=False,
    )
    if vin_snippet_cache_dir is not None:
        cache_cfg.vin_snippet_cache = VinSnippetCacheConfig(
            cache_dir=vin_snippet_cache_dir,
        )

    return VinDataModuleConfig(
        source=VinOracleCacheDatasetConfig(
            cache=cache_cfg,
            train_split=split,
            val_split=split,
        ),
        use_train_as_val=True,
        shuffle=False,
        num_workers=0,
        batch_size=batch_size,
        persistent_workers=False,
        verbosity=Verbosity.QUIET,
    )


def test_vin_snippet_cache_matches_offline_cache_via_datamodule_single(
    tmp_path: Path,
) -> None:
    """Ensure VinSnippetCache matches live EFM loading (single sample)."""
    paths = PathConfig()
    oracle_cache_dir = paths.offline_cache_dir
    _skip_if_missing_oracle_cache(oracle_cache_dir)

    split = "train"
    limit = 1

    live_cfg = _make_offline_datamodule_cfg(
        cache_dir=oracle_cache_dir,
        vin_snippet_cache_dir=None,
        split=split,
        limit=limit,
        batch_size=None,
    )
    try:
        live_batch = _first_train_batch(live_cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"Failed to load live EFM snippet for oracle cache sample: {exc}")

    vin_cache_dir = tmp_path / "vin_snippet_cache"
    writer_cfg = VinSnippetCacheWriterConfig(
        cache=VinSnippetCacheConfig(cache_dir=vin_cache_dir),
        source_cache=OracleRriCacheConfig(cache_dir=oracle_cache_dir),
        split=split,
        max_samples=limit,
        semidense_max_points=None,
        overwrite=True,
        map_location="cpu",
        verbosity=Verbosity.QUIET,
    )
    writer_cfg.setup_target().run()

    cached_cfg = _make_offline_datamodule_cfg(
        cache_dir=oracle_cache_dir,
        vin_snippet_cache_dir=vin_cache_dir,
        split=split,
        limit=limit,
        batch_size=None,
    )
    cached_batch = _first_train_batch(cached_cfg)

    if live_batch.scene_id != cached_batch.scene_id:
        raise AssertionError("scene_id differs between live and cached batches.")
    if live_batch.snippet_id != cached_batch.snippet_id:
        raise AssertionError("snippet_id differs between live and cached batches.")

    live_view = _as_vin_snippet(live_batch)
    cached_view = _as_vin_snippet(cached_batch)

    torch.testing.assert_close(
        live_view.t_world_rig.tensor(),
        cached_view.t_world_rig.tensor(),
        rtol=0.0,
        atol=0.0,
    )
    assert int(cached_view.points_world.shape[0]) == VIN_SNIPPET_PAD_POINTS
    live_len = int(live_view.lengths.reshape(-1)[0].item())
    cached_len = int(cached_view.lengths.reshape(-1)[0].item())
    assert cached_len <= VIN_SNIPPET_PAD_POINTS
    if live_len <= VIN_SNIPPET_PAD_POINTS:
        torch.testing.assert_close(
            live_view.points_world,
            cached_view.points_world,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    else:
        assert cached_len == VIN_SNIPPET_PAD_POINTS


def test_vin_snippet_cache_matches_offline_cache_via_datamodule_batched(
    tmp_path: Path,
) -> None:
    """Ensure VinSnippetCache and live EFM loading match under VIN batching."""
    paths = PathConfig()
    oracle_cache_dir = paths.offline_cache_dir
    _skip_if_missing_oracle_cache(oracle_cache_dir)

    split = "train"
    batch_size = 2
    limit = batch_size

    live_cfg = _make_offline_datamodule_cfg(
        cache_dir=oracle_cache_dir,
        vin_snippet_cache_dir=None,
        split=split,
        limit=limit,
        batch_size=batch_size,
    )
    try:
        live_batch = _first_train_batch(live_cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(
            f"Failed to load live EFM snippet batches for oracle cache samples: {exc}",
        )

    vin_cache_dir = tmp_path / "vin_snippet_cache"
    writer_cfg = VinSnippetCacheWriterConfig(
        cache=VinSnippetCacheConfig(cache_dir=vin_cache_dir),
        source_cache=OracleRriCacheConfig(cache_dir=oracle_cache_dir),
        split=split,
        max_samples=limit,
        semidense_max_points=None,
        overwrite=True,
        map_location="cpu",
        verbosity=Verbosity.QUIET,
    )
    writer_cfg.setup_target().run()

    cached_cfg = _make_offline_datamodule_cfg(
        cache_dir=oracle_cache_dir,
        vin_snippet_cache_dir=vin_cache_dir,
        split=split,
        limit=limit,
        batch_size=batch_size,
    )
    cached_batch = _first_train_batch(cached_cfg)

    if live_batch.scene_id != cached_batch.scene_id:
        raise AssertionError("scene_id differs between live and cached batches.")
    if live_batch.snippet_id != cached_batch.snippet_id:
        raise AssertionError("snippet_id differs between live and cached batches.")

    live_view = _as_vin_snippet(live_batch)
    cached_view = _as_vin_snippet(cached_batch)

    if live_view.points_world.ndim != _BATCHED_POINTS_NDIM:
        raise AssertionError("Expected live points_world to be batched (ndim=3).")
    if cached_view.points_world.ndim != _BATCHED_POINTS_NDIM:
        raise AssertionError("Expected cached points_world to be batched (ndim=3).")
    if int(live_view.points_world.shape[0]) != batch_size:
        raise AssertionError("Unexpected batch dimension for live points_world.")
    if int(cached_view.points_world.shape[0]) != batch_size:
        raise AssertionError("Unexpected batch dimension for cached points_world.")
    if int(live_view.points_world.shape[-1]) != _VIN_POINTS_DIM:
        raise AssertionError("Expected live points_world to have last dim == 4.")
    if int(cached_view.points_world.shape[-1]) != _VIN_POINTS_DIM:
        raise AssertionError("Expected cached points_world to have last dim == 4.")

    torch.testing.assert_close(
        live_view.t_world_rig.tensor(),
        cached_view.t_world_rig.tensor(),
        rtol=0.0,
        atol=0.0,
    )
    live_lengths = live_view.lengths.to(dtype=torch.int64).reshape(-1)
    cached_lengths = cached_view.lengths.to(dtype=torch.int64).reshape(-1)
    assert torch.all(cached_lengths <= VIN_SNIPPET_PAD_POINTS)
    if torch.all(live_lengths <= VIN_SNIPPET_PAD_POINTS):
        torch.testing.assert_close(
            live_view.points_world,
            cached_view.points_world,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

"""Tests for train/val index splitting in the offline cache dataset."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path

import pytest


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


def _write_index(path: Path, entries: list[dict[str, str]]) -> None:
    payload = "\n".join(json.dumps(entry) for entry in entries)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _index_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        keys.append(json.loads(line)["key"])
    return keys


def _build_entries(count: int) -> list[dict[str, str]]:
    return [
        {
            "key": f"key_{idx}",
            "scene_id": "scene",
            "snippet_id": f"snippet_{idx}",
            "path": f"samples/key_{idx}.pt",
        }
        for idx in range(count)
    ]


def _unwrap_base_dataset(dataset: object) -> object:
    return getattr(dataset, "_base", dataset)


def test_rebuild_cache_index_random_split(tmp_path: Path) -> None:
    """Rebuild split indices using a deterministic RNG seed."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    rebuild_cache_index = offline_mod.rebuild_cache_index

    cache_dir = tmp_path / "cache"
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True)

    keys = []
    for idx in range(10):
        key = f"ASE_NBV_SNIPPET_scene_snip{idx:02d}_hash"
        keys.append(key)
        (samples_dir / f"{key}.pt").write_text("x", encoding="utf-8")

    seed = 1234
    rebuild_cache_index(cache_dir=cache_dir, train_val_split=0.3, rng_seed=seed)

    train_path = cache_dir / "train_index.jsonl"
    val_path = cache_dir / "val_index.jsonl"
    train_keys = _index_keys(train_path)
    val_keys = _index_keys(val_path)

    rng = random.Random(seed)  # noqa: S311
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    expected_val = shuffled[:3]
    expected_train = shuffled[3:]

    assert val_keys == expected_val  # noqa: S101
    assert train_keys == expected_train  # noqa: S101


def test_cache_split_creates_train_val_indices(tmp_path: Path) -> None:
    """Create train/val index files and verify the split sizes."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "samples").mkdir()
    _write_dummy_metadata(cache_dir)

    total_entries = 10
    val_fraction = 0.2
    entries = _build_entries(total_entries)
    _write_index(cache_dir / "index.jsonl", entries)

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    train_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=val_fraction,
    )
    val_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="val",
        train_val_split=val_fraction,
    )
    train_ds = train_cfg.setup_target()
    val_ds = val_cfg.setup_target()

    expected_val = round(total_entries * val_fraction)
    expected_train = total_entries - expected_val
    assert len(train_ds) == expected_train  # noqa: S101
    assert len(val_ds) == expected_val  # noqa: S101

    train_keys = set(_index_keys(cache_cfg.train_index_path))
    val_keys = set(_index_keys(cache_cfg.val_index_path))
    base_keys = {entry["key"] for entry in entries}

    assert train_keys.isdisjoint(val_keys)  # noqa: S101
    assert train_keys | val_keys == base_keys  # noqa: S101


def test_cache_split_preserves_existing_assignments(tmp_path: Path) -> None:
    """Keep existing train/val assignments when new entries are appended."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "samples").mkdir()
    _write_dummy_metadata(cache_dir)

    val_fraction = 0.2
    entries = _build_entries(5)
    _write_index(cache_dir / "index.jsonl", entries)

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    train_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=val_fraction,
    )
    val_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="val",
        train_val_split=val_fraction,
    )
    train_cfg.setup_target()
    val_cfg.setup_target()

    val_keys_before = _index_keys(cache_cfg.val_index_path)

    new_entry = _build_entries(6)[-1]
    with (cache_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(new_entry) + "\n")

    OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=0.2,
    ).setup_target()

    val_keys_after = _index_keys(cache_cfg.val_index_path)

    assert val_keys_before == val_keys_after  # noqa: S101


def test_cache_split_real_data() -> None:
    """Exercise train/val split on an existing offline cache."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / ".data" / "oracle_rri_cache"
    if not (cache_dir / "index.jsonl").exists():
        pytest.skip("Missing real offline cache index.jsonl")
    if not (cache_dir / "metadata.json").exists():
        pytest.skip("Missing real offline cache metadata.json")

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    train_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=0.2,
    )
    train_ds = train_cfg.setup_target()
    if len(train_ds) == 0:
        pytest.skip("Offline cache contains no samples")
    sample = train_ds[0]
    assert sample.rri.rri.numel() > 0  # noqa: S101


def test_datamodule_applies_cache_split(tmp_path: Path) -> None:
    """VinDataModule should honor explicit train/val cache splits."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    dm_mod = importlib.import_module("oracle_rri.lightning.lit_datamodule")
    datasets_mod = importlib.import_module("oracle_rri.data.vin_oracle_datasets")
    utils_mod = importlib.import_module("oracle_rri.utils")

    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806
    VinOracleCacheDatasetConfig = datasets_mod.VinOracleCacheDatasetConfig  # noqa: N806
    VinDataModuleConfig = dm_mod.VinDataModuleConfig  # noqa: N806
    VinDataModule = dm_mod.VinDataModule  # noqa: N806
    Stage = utils_mod.Stage  # noqa: N806

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "samples").mkdir()
    _write_dummy_metadata(cache_dir)
    _write_index(cache_dir / "index.jsonl", _build_entries(6))

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    base_cache_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="all",
        train_val_split=0.25,
    )
    dm_cfg = VinDataModuleConfig(
        source=VinOracleCacheDatasetConfig(
            cache=base_cache_cfg,
            train_split="train",
            val_split="val",
        ),
        num_workers=0,
    )
    dm = VinDataModule(dm_cfg)
    dm.setup(stage=Stage.TRAIN)

    train_ds = dm.train_dataloader().dataset
    val_ds = dm.val_dataloader().dataset
    assert train_ds is not None  # noqa: S101
    assert val_ds is not None  # noqa: S101
    train_base = _unwrap_base_dataset(train_ds)
    val_base = _unwrap_base_dataset(val_ds)
    assert getattr(getattr(train_base, "config", None), "split", None) == "train"  # noqa: S101
    assert getattr(getattr(val_base, "config", None), "split", None) == "val"  # noqa: S101


def test_cache_dataset_include_snippet_real_data() -> None:
    """Load an EFM snippet from the offline cache when enabled."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / ".data" / "oracle_rri_cache"
    if not (cache_dir / "index.jsonl").exists():
        pytest.skip("Missing real offline cache index.jsonl")
    if not (cache_dir / "metadata.json").exists():
        pytest.skip("Missing real offline cache metadata.json")
    if not (repo_root / ".data" / "ase_efm").exists():
        pytest.skip("Missing ASE EFM data directory")

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    cache_ds_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=0.2,
        include_efm_snippet=True,
        include_gt_mesh=False,
    )
    cache_ds = cache_ds_cfg.setup_target()
    if len(cache_ds) == 0:
        pytest.skip("Offline cache contains no samples")
    sample = cache_ds[0]
    assert sample.efm_snippet_view is not None  # noqa: S101


def test_cache_dataset_returns_vin_batch_real_data() -> None:
    """Return VIN batch format directly from the cache dataset."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / ".data" / "oracle_rri_cache"
    if not (cache_dir / "index.jsonl").exists():
        pytest.skip("Missing real offline cache index.jsonl")
    if not (cache_dir / "metadata.json").exists():
        pytest.skip("Missing real offline cache metadata.json")

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    cache_ds_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=0.2,
        return_format="vin_batch",
    )
    cache_ds = cache_ds_cfg.setup_target()
    if len(cache_ds) == 0:
        pytest.skip("Offline cache contains no samples")
    batch = cache_ds[0]
    assert batch.rri.numel() > 0  # noqa: S101


def test_cache_dataset_simplification(tmp_path: Path) -> None:
    """Simplification should reduce the exposed length."""
    offline_mod = importlib.import_module("oracle_rri.data.offline_cache")
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "samples").mkdir()
    _write_dummy_metadata(cache_dir)

    total_entries = 10
    val_fraction = 0.2
    entries = _build_entries(total_entries)
    _write_index(cache_dir / "index.jsonl", entries)

    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir)
    ds_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        split="train",
        train_val_split=val_fraction,
        simplification=0.3,
    )
    ds = ds_cfg.setup_target()
    train_count = total_entries - round(total_entries * val_fraction)
    expected_len = int(train_count * 0.3)
    assert len(ds) == expected_len  # noqa: S101

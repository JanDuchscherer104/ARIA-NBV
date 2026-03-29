"""Tests for VinDataModule source-config validation."""

from __future__ import annotations

import pytest
from oracle_rri.data.vin_oracle_datasets import (
    VinOracleCacheDatasetConfig,
    VinOracleOnlineDatasetConfig,
)
from oracle_rri.lightning.lit_datamodule import VinDataModuleConfig


def test_source_online_requires_single_worker() -> None:
    """Online sources should reject multiprocessing workers."""
    with pytest.raises(ValueError, match="not multiprocess-safe"):
        VinDataModuleConfig(
            source=VinOracleOnlineDatasetConfig(),
            num_workers=1,
            use_train_as_val=True,
        )


def test_source_cache_allows_batching() -> None:
    """Offline cache sources can opt into batching."""
    batch_size = 2
    cfg = VinDataModuleConfig(
        source=VinOracleCacheDatasetConfig(),
        batch_size=batch_size,
        use_train_as_val=True,
    )
    assert cfg.batch_size == batch_size  # noqa: S101


def test_source_online_rejects_batching() -> None:
    """Online sources cannot be batched."""
    with pytest.raises(ValueError, match="batch_size can only be used"):
        VinDataModuleConfig(
            source=VinOracleOnlineDatasetConfig(),
            batch_size=2,
            use_train_as_val=True,
        )

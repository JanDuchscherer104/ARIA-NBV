"""Legacy compatibility wrapper for the canonical oracle-cache surface.

The active oracle-cache implementation now lives in
``aria_nbv.data_handling.oracle_cache`` and
``aria_nbv.data_handling.oracle_cache_datasets``. This legacy module keeps the
historical ``aria_nbv.data.offline_cache`` import path stable for tests and
remaining callers under ``aria_nbv.data``.
"""

from __future__ import annotations

from pathlib import Path

from ..configs import PathConfig
from ..data_handling.cache_contracts import (
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
)
from ..data_handling.offline_cache_store import snapshot_config, snapshot_dataset_config
from ..data_handling.oracle_cache import (
    CACHE_VERSION,
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
    build_cache_payload,
    decode_backbone,
    decode_candidate_pcs,
    decode_candidates,
    decode_depths,
    decode_rri,
)
from ..data_handling.oracle_cache_datasets import (
    OracleRriCacheDataset,
    OracleRriCacheVinDataset,
)
from .efm_dataset import AseEfmDatasetConfig
from .efm_snippet_loader import EfmSnippetLoader


def rebuild_cache_index(
    *,
    cache_dir: Path,
    train_val_split: float | None = None,
    rng_seed: int | None = None,
) -> int:
    """Rebuild ``index.jsonl`` and train/validation splits from sample files."""
    cache_cfg = OracleRriCacheConfig(cache_dir=cache_dir, paths=PathConfig())
    return cache_cfg.rebuild_index(
        train_val_split=train_val_split,
        rng_seed=rng_seed,
    )


__all__ = [
    "AseEfmDatasetConfig",
    "CACHE_VERSION",
    "EfmSnippetLoader",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "OracleRriCacheVinDataset",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "build_cache_payload",
    "decode_backbone",
    "decode_candidate_pcs",
    "decode_candidates",
    "decode_depths",
    "decode_rri",
    "rebuild_cache_index",
    "snapshot_config",
    "snapshot_dataset_config",
]

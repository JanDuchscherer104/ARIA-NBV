"""Grouped legacy oracle/VIN cache exports for explicit migration-period imports.

NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION:
Import legacy cache functionality from this module when code must still depend
on the old oracle-cache / VIN-snippet-cache stack. This keeps that dependency
grep-visible and separate from the canonical ``aria_nbv.data_handling`` root.
"""

from __future__ import annotations

from ._legacy_offline_cache_coverage import (
    CacheCoverageReport,
    SceneCoverage,
    compute_cache_coverage,
    expand_tar_urls,
    read_cache_index_entries,
    scan_dataset_snippets,
    scan_tar_sample_keys,
    snippets_by_scene,
)
from ._legacy_offline_cache_store import extract_snippet_token, read_oracle_cache_metadata
from ._legacy_oracle_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
    OracleRriCacheVinDataset,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
    rebuild_cache_index,
    rebuild_oracle_cache_index,
    repair_oracle_cache_indices,
)
from ._legacy_vin_cache import (
    VIN_SNIPPET_CACHE_VERSION,
    VIN_SNIPPET_PAD_POINTS,
    VinSnippetCacheBuildResult,
    VinSnippetCacheConfig,
    VinSnippetCacheDataset,
    VinSnippetCacheDatasetConfig,
    VinSnippetCacheEntry,
    VinSnippetCacheMetadata,
    VinSnippetCacheWriter,
    VinSnippetCacheWriterConfig,
    read_vin_snippet_cache_metadata,
    rebuild_vin_snippet_cache_index,
    repair_vin_snippet_cache_index,
)

__all__ = [
    "CacheCoverageReport",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "OracleRriCacheVinDataset",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "SceneCoverage",
    "VIN_SNIPPET_CACHE_VERSION",
    "VIN_SNIPPET_PAD_POINTS",
    "VinSnippetCacheBuildResult",
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "compute_cache_coverage",
    "expand_tar_urls",
    "extract_snippet_token",
    "read_cache_index_entries",
    "read_oracle_cache_metadata",
    "read_vin_snippet_cache_metadata",
    "rebuild_cache_index",
    "rebuild_oracle_cache_index",
    "rebuild_vin_snippet_cache_index",
    "repair_oracle_cache_indices",
    "repair_vin_snippet_cache_index",
    "scan_dataset_snippets",
    "scan_tar_sample_keys",
    "snippets_by_scene",
]

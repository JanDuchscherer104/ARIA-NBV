"""Legacy compatibility wrapper for the canonical VIN snippet cache surface.

The active implementation lives in :mod:`aria_nbv.data_handling.vin_cache`.
This module remains only to preserve legacy import paths used by tests, the
CLI, and app surfaces that still reference :mod:`aria_nbv.data`.
"""

from __future__ import annotations

from ..data_handling.vin_cache import (
    VIN_SNIPPET_CACHE_VERSION,
    VIN_SNIPPET_PAD_POINTS,
    VinSnippetCacheBuildDataset,
    VinSnippetCacheBuildResult,
    VinSnippetCacheConfig,
    VinSnippetCacheDataset,
    VinSnippetCacheDatasetConfig,
    VinSnippetCacheEntry,
    VinSnippetCacheMetadata,
    VinSnippetCacheWriter,
    VinSnippetCacheWriterConfig,
    migrate_vin_snippet_cache_inplace,
    read_vin_snippet_cache_metadata,
    rebuild_vin_snippet_cache_index,
    repair_vin_snippet_cache_index,
)

__all__ = [
    "VIN_SNIPPET_CACHE_VERSION",
    "VIN_SNIPPET_PAD_POINTS",
    "VinSnippetCacheBuildDataset",
    "VinSnippetCacheBuildResult",
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "migrate_vin_snippet_cache_inplace",
    "read_vin_snippet_cache_metadata",
    "rebuild_vin_snippet_cache_index",
    "repair_vin_snippet_cache_index",
]

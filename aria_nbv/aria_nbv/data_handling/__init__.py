"""Public root API for raw snippets, VIN runtime types, and offline storage.

This package is the stable public contract for the training-data core:

- raw ASE/EFM snippets and typed views,
- VIN-facing runtime helpers and batch types,
- the immutable VIN offline dataset format and writer, and
- temporary compatibility exports for the legacy oracle-cache and VIN-snippet
  cache surfaces during migration.

Code outside ``aria_nbv.data_handling`` should import from this module rather
than reaching into package submodules directly. Low-level shard handles,
serialization helpers, and migration plumbing stay internal even when they are
implemented in sibling modules inside this package.
"""

# ruff: noqa: I001

from __future__ import annotations

from ._raw import (
    AseEfmDataset,
    AseEfmDatasetConfig,
    EfmCameraView,
    EfmGTView,
    EfmObbView,
    EfmPointsView,
    EfmSnippetView,
    EfmTrajectoryView,
    VinSnippetView,
    infer_semidense_bounds,
    is_efm_snippet_view_instance,
    is_vin_snippet_view_instance,
)
from ._vin_runtime import (
    DEFAULT_VIN_SNIPPET_PAD_POINTS,
    VinOnlineDatasetConfig,
    VinOracleBatch,
    VinOracleDatasetBase,
    VinOracleOnlineDataset,
    VinOracleOnlineDatasetConfig,
    build_vin_snippet_view,
    empty_vin_snippet,
)
from ._offline_format import (
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
)
from ._offline_store import OFFLINE_DATASET_VERSION, VinOfflineStoreConfig
from ._offline_dataset import (
    VinOfflineDataset,
    VinOfflineDatasetConfig,
    VinOfflineSample,
)
from ._offline_writer import (
    VinOfflineWriter,
    VinOfflineWriterConfig,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from ._migration import (
    migrate_legacy_offline_data,
    scan_legacy_offline_data,
    verify_migrated_offline_data,
)
from .mesh_cache import MeshProcessSpec, ProcessedMesh, load_or_process_mesh
from .oracle_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
    OracleRriCacheSample,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
    repair_oracle_cache_indices,
)
from .oracle_cache_datasets import OracleRriCacheDataset, OracleRriCacheVinDataset
from .vin_cache import (
    VIN_SNIPPET_CACHE_VERSION,
    VIN_SNIPPET_PAD_POINTS,
    VinSnippetCacheConfig,
    VinSnippetCacheDataset,
    VinSnippetCacheDatasetConfig,
    VinSnippetCacheWriter,
    VinSnippetCacheWriterConfig,
    read_vin_snippet_cache_metadata,
    rebuild_vin_snippet_cache_index,
    repair_vin_snippet_cache_index,
)
from .vin_oracle_datasets import (
    VinDatasetSourceConfig,
    VinOfflineSourceConfig,
    VinOracleCacheDatasetConfig,
    VinOracleDatasetConfig,
)

__all__ = [
    "AseEfmDataset",
    "AseEfmDatasetConfig",
    "DEFAULT_VIN_SNIPPET_PAD_POINTS",
    "EfmCameraView",
    "EfmGTView",
    "EfmObbView",
    "EfmPointsView",
    "EfmSnippetView",
    "EfmTrajectoryView",
    "MeshProcessSpec",
    "OFFLINE_DATASET_VERSION",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheSample",
    "OracleRriCacheVinDataset",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "ProcessedMesh",
    "VIN_SNIPPET_CACHE_VERSION",
    "VIN_SNIPPET_PAD_POINTS",
    "VinDatasetSourceConfig",
    "VinOfflineDataset",
    "VinOfflineDatasetConfig",
    "VinOfflineIndexRecord",
    "VinOfflineManifest",
    "VinOfflineMaterializedBlocks",
    "VinOfflineSample",
    "VinOfflineSourceConfig",
    "VinOfflineStoreConfig",
    "VinOfflineWriter",
    "VinOfflineWriterConfig",
    "VinOnlineDatasetConfig",
    "VinOracleBatch",
    "VinOracleCacheDatasetConfig",
    "VinOracleDatasetBase",
    "VinOracleDatasetConfig",
    "VinOracleOnlineDataset",
    "VinOracleOnlineDatasetConfig",
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "VinSnippetView",
    "build_vin_snippet_view",
    "empty_vin_snippet",
    "flush_prepared_samples_to_shard",
    "infer_semidense_bounds",
    "is_efm_snippet_view_instance",
    "is_vin_snippet_view_instance",
    "load_or_process_mesh",
    "migrate_legacy_offline_data",
    "prepare_vin_offline_sample",
    "read_vin_snippet_cache_metadata",
    "rebuild_vin_snippet_cache_index",
    "repair_oracle_cache_indices",
    "repair_vin_snippet_cache_index",
    "scan_legacy_offline_data",
    "verify_migrated_offline_data",
]

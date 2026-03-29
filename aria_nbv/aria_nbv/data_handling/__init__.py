"""Public root API for raw snippets, VIN runtime types, and offline storage.

This package is the stable public contract for the training-data core:

- raw ASE/EFM snippets and typed views,
- VIN-facing runtime helpers and batch types,
- the new immutable VIN offline dataset format and writer, and
- temporary compatibility exports for the legacy oracle-cache and VIN-snippet
  cache surfaces during migration.

Code outside ``aria_nbv.data_handling`` should import from this module rather
than reaching into package submodules directly.
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
    EfmSnippetLoader,
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
    collapse_vin_points,
    empty_vin_snippet,
    pad_vin_points,
    vin_snippet_cache_config_hash,
)
from ._offline_format import (
    VinOfflineBlockSpec,
    VinOfflineCounterfactuals,
    VinOfflineIndexRecord,
    VinOfflineManifest,
    VinOfflineMaterializedBlocks,
    VinOfflineShardSpec,
)
from ._offline_store import OFFLINE_DATASET_VERSION, OpenedShard, VinOfflineStoreConfig, VinOfflineStoreReader
from ._offline_dataset import (
    VinOfflineDataset,
    VinOfflineDatasetConfig,
    VinOfflineOracleBlock,
    VinOfflineSample,
)
from ._offline_writer import (
    PreparedVinOfflineSample,
    VinOfflineWriter,
    VinOfflineWriterConfig,
    flush_prepared_samples_to_shard,
    prepare_vin_offline_sample,
)
from ._migration import (
    LegacyOfflinePlan,
    LegacyOfflineRecord,
    finalize_migrated_store,
    prepare_legacy_records,
    scan_legacy_offline_data,
    verify_migrated_offline_data,
)
from .cache_contracts import (
    OracleRriCacheEntry,
    OracleRriCacheMetadata,
    OracleRriCacheSample,
    VinSnippetCacheEntry,
    VinSnippetCacheMetadata,
)
from .mesh_cache import MeshProcessSpec, ProcessedMesh, load_or_process_mesh
from .oracle_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheVinDataset,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
    rebuild_cache_index,
    rebuild_oracle_cache_index,
    repair_oracle_cache_indices,
)
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
    "EfmSnippetLoader",
    "EfmSnippetView",
    "EfmTrajectoryView",
    "LegacyOfflinePlan",
    "LegacyOfflineRecord",
    "MeshProcessSpec",
    "OFFLINE_DATASET_VERSION",
    "OpenedShard",
    "OracleRriCacheConfig",
    "OracleRriCacheDataset",
    "OracleRriCacheDatasetConfig",
    "OracleRriCacheEntry",
    "OracleRriCacheMetadata",
    "OracleRriCacheSample",
    "OracleRriCacheVinDataset",
    "OracleRriCacheWriter",
    "OracleRriCacheWriterConfig",
    "PreparedVinOfflineSample",
    "ProcessedMesh",
    "VIN_SNIPPET_CACHE_VERSION",
    "VIN_SNIPPET_PAD_POINTS",
    "VinDatasetSourceConfig",
    "VinOfflineBlockSpec",
    "VinOfflineCounterfactuals",
    "VinOfflineDataset",
    "VinOfflineDatasetConfig",
    "VinOfflineIndexRecord",
    "VinOfflineManifest",
    "VinOfflineMaterializedBlocks",
    "VinOfflineOracleBlock",
    "VinOfflineSample",
    "VinOfflineShardSpec",
    "VinOfflineSourceConfig",
    "VinOfflineStoreConfig",
    "VinOfflineStoreReader",
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
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "VinSnippetView",
    "build_vin_snippet_view",
    "collapse_vin_points",
    "empty_vin_snippet",
    "finalize_migrated_store",
    "flush_prepared_samples_to_shard",
    "infer_semidense_bounds",
    "is_efm_snippet_view_instance",
    "is_vin_snippet_view_instance",
    "load_or_process_mesh",
    "pad_vin_points",
    "prepare_legacy_records",
    "prepare_vin_offline_sample",
    "read_vin_snippet_cache_metadata",
    "rebuild_cache_index",
    "rebuild_oracle_cache_index",
    "rebuild_vin_snippet_cache_index",
    "repair_oracle_cache_indices",
    "repair_vin_snippet_cache_index",
    "scan_legacy_offline_data",
    "verify_migrated_offline_data",
    "vin_snippet_cache_config_hash",
]

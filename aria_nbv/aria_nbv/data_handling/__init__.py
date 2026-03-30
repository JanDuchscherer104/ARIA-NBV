"""Canonical public API for raw snippets, VIN runtime types, and offline storage.

This package root intentionally exposes the supported *canonical* data-handling
surface:

- raw ASE/EFM snippets and typed views,
- VIN-facing runtime helpers and batch types,
- the immutable VIN offline dataset format and writer, and
- migration entry points for converting legacy cache artifacts.

Legacy oracle-cache, VIN-snippet-cache, and coverage utilities now live behind
dedicated ``_legacy_*`` compatibility modules. Old direct submodule imports
such as ``aria_nbv.data_handling.oracle_cache`` still resolve, but new code
should not rely on them through the package root.
"""

from __future__ import annotations

from importlib import import_module

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

_LAZY_EXPORTS = {
    "DEFAULT_VIN_SNIPPET_PAD_POINTS": "._vin_runtime",
    "MeshProcessSpec": ".mesh_cache",
    "OFFLINE_DATASET_VERSION": "._offline_store",
    "ProcessedMesh": ".mesh_cache",
    "VinDatasetSourceConfig": "._vin_sources",
    "VinOfflineDataset": "._offline_dataset",
    "VinOfflineDatasetConfig": "._offline_dataset",
    "VinOfflineIndexRecord": "._offline_format",
    "VinOfflineManifest": "._offline_format",
    "VinOfflineMaterializedBlocks": "._offline_format",
    "VinOfflineSample": "._offline_dataset",
    "VinOfflineSourceConfig": "._vin_sources",
    "VinOfflineStoreConfig": "._offline_store",
    "VinOfflineWriter": "._offline_writer",
    "VinOfflineWriterConfig": "._offline_writer",
    "VinOnlineDatasetConfig": "._vin_runtime",
    "VinOracleBatch": "._vin_runtime",
    "VinOracleDatasetBase": ".vin_oracle_types",
    "VinOracleOnlineDataset": "._vin_runtime",
    "VinOracleOnlineDatasetConfig": "._vin_runtime",
    "build_vin_snippet_view": "._vin_runtime",
    "empty_vin_snippet": "._vin_runtime",
    "flush_prepared_samples_to_shard": "._offline_writer",
    "load_or_process_mesh": ".mesh_cache",
    "migrate_legacy_offline_data": "._migration",
    "prepare_vin_offline_sample": "._offline_writer",
    "scan_legacy_offline_data": "._migration",
    "verify_migrated_offline_data": "._migration",
}


def __getattr__(name: str) -> object:
    """Lazily resolve non-raw exports to avoid package-root import cycles."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


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
    "ProcessedMesh",
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
    "VinOracleDatasetBase",
    "VinOracleOnlineDataset",
    "VinOracleOnlineDatasetConfig",
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
    "scan_legacy_offline_data",
    "verify_migrated_offline_data",
]

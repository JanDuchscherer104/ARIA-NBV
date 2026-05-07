"""Canonical public API for raw snippets, VIN runtime types, and offline storage.

This package root intentionally exposes the supported *canonical* data-handling
surface:

- raw ASE/EFM snippets and typed views,
- VIN-facing runtime helpers and batch types,
- the immutable VIN offline dataset format and writer.
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
    "ActorVisibleTargetSelector": "._target_selection",
    "DEFAULT_VIN_SNIPPET_PAD_POINTS": "._vin_runtime",
    "CompactObbBlock": ".vin_oracle_types",
    "CompactTrajectoryBlock": ".vin_oracle_types",
    "MeshProcessSpec": ".mesh_cache",
    "OFFLINE_DATASET_VERSION": "._offline_store",
    "OfflineVisualInventory": "._offline_visual_inventory",
    "OfflineVisualInventoryError": "._offline_visual_inventory",
    "ProcessedMesh": ".mesh_cache",
    "NumericSummary": "._offline_diagnostics",
    "RolloutZarrStoreConfig": "._rollout_zarr_store",
    "RolloutZarrStoreReader": "._rollout_zarr_store",
    "RolloutZarrStoreWriter": "._rollout_zarr_store",
    "RolloutZarrValidationResult": "._rollout_zarr_store",
    "RolloutZarrWriteResult": "._rollout_zarr_store",
    "TARGET_INVALID_REASON_CODES": "._target_selection",
    "TARGET_INVALID_REASON_VERSION": "._target_selection",
    "TargetCandidateRow": "._target_selection",
    "TargetSelectionPolicy": "._target_selection",
    "TargetSelectionResult": "._target_selection",
    "TargetSelectorConfig": "._target_selection",
    "TargetSourceMode": "._target_selection",
    "VinDatasetSourceConfig": "._vin_sources",
    "VinOfflineBackboneDiagnostic": "._offline_diagnostics",
    "VinOfflineBlockDiagnostic": "._offline_diagnostics",
    "VinOfflineCoverageSceneDiagnostic": "._offline_diagnostics",
    "VinOfflineCoverageStats": "._offline_diagnostics",
    "VinOfflineDataset": "._offline_dataset",
    "VinOfflineDatasetConfig": "._offline_dataset",
    "VinOfflineDatasetStats": "._offline_diagnostics",
    "VinOfflineIndexRecord": "._offline_format",
    "VinOfflineManifest": "._offline_format",
    "VinOfflineMaterializedBlocks": "._offline_format",
    "VinOfflineMemoryDiagnostic": "._offline_diagnostics",
    "VinOfflineSample": "._offline_dataset",
    "VinOfflineSampleDiagnostic": "._offline_diagnostics",
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
    "collect_vin_offline_dataset_coverage": "._offline_diagnostics",
    "collect_vin_offline_dataset_stats": "._offline_diagnostics",
    "collect_offline_visual_inventory": "._offline_visual_inventory",
    "empty_vin_snippet": "._vin_runtime",
    "flush_prepared_samples_to_shard": "._offline_writer",
    "load_or_process_mesh": ".mesh_cache",
    "prepare_vin_offline_sample": "._offline_writer",
    "validate_rollout_zarr_store": "._rollout_zarr_store",
    "write_rollout_zarr_store": "._rollout_zarr_store",
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
    "ActorVisibleTargetSelector",
    "AseEfmDataset",
    "AseEfmDatasetConfig",
    "CompactObbBlock",
    "CompactTrajectoryBlock",
    "DEFAULT_VIN_SNIPPET_PAD_POINTS",
    "EfmCameraView",
    "EfmGTView",
    "EfmObbView",
    "EfmPointsView",
    "EfmSnippetView",
    "EfmTrajectoryView",
    "MeshProcessSpec",
    "NumericSummary",
    "OFFLINE_DATASET_VERSION",
    "OfflineVisualInventory",
    "OfflineVisualInventoryError",
    "ProcessedMesh",
    "RolloutZarrStoreConfig",
    "RolloutZarrStoreReader",
    "RolloutZarrStoreWriter",
    "RolloutZarrValidationResult",
    "RolloutZarrWriteResult",
    "TARGET_INVALID_REASON_CODES",
    "TARGET_INVALID_REASON_VERSION",
    "TargetCandidateRow",
    "TargetSelectionPolicy",
    "TargetSelectionResult",
    "TargetSelectorConfig",
    "TargetSourceMode",
    "VinDatasetSourceConfig",
    "VinOfflineDataset",
    "VinOfflineBackboneDiagnostic",
    "VinOfflineBlockDiagnostic",
    "VinOfflineCoverageSceneDiagnostic",
    "VinOfflineCoverageStats",
    "VinOfflineDatasetConfig",
    "VinOfflineDatasetStats",
    "VinOfflineIndexRecord",
    "VinOfflineManifest",
    "VinOfflineMaterializedBlocks",
    "VinOfflineMemoryDiagnostic",
    "VinOfflineSample",
    "VinOfflineSampleDiagnostic",
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
    "collect_vin_offline_dataset_coverage",
    "collect_vin_offline_dataset_stats",
    "collect_offline_visual_inventory",
    "empty_vin_snippet",
    "flush_prepared_samples_to_shard",
    "infer_semidense_bounds",
    "is_efm_snippet_view_instance",
    "is_vin_snippet_view_instance",
    "load_or_process_mesh",
    "prepare_vin_offline_sample",
    "validate_rollout_zarr_store",
    "write_rollout_zarr_store",
]

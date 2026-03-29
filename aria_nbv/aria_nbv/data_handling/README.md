# `aria_nbv.data_handling`

`aria_nbv.data_handling` is the core data package for Aria-NBV training and
offline diagnostics. It owns:

- the raw ASE/EFM snippet dataset,
- the canonical `EfmSnippetView -> VinSnippetView` adapter,
- the model-facing `VinOracleBatch` contract,
- the new immutable VIN offline dataset format, and
- temporary migration and compatibility surfaces for the legacy oracle and VIN caches.

Code outside this package should import from `aria_nbv.data_handling` rather
than from `aria_nbv.data_handling.*` submodules directly.

## Design goals

- Keep the raw ASE/EFM contract unchanged.
- Have one canonical VIN snippet representation for training.
- Replace per-sample `torch.load` offline datasets with immutable indexed shards.
- Keep training-critical tensors in zarr-backed fixed arrays for fast
  multi-worker random access.
- Keep richer diagnostic payloads lazy and optional.
- Treat legacy oracle-cache and VIN-snippet-cache code as temporary
  compatibility layers while the new store becomes the primary offline path.

## Public surface

The package root `aria_nbv.data_handling` exports the supported public API.
The most important groups are:

- Raw source layer
  - `AseEfmDatasetConfig`, `AseEfmDataset`
  - `EfmSnippetView`, `EfmCameraView`, `EfmTrajectoryView`,
    `EfmPointsView`, `EfmObbView`, `EfmGTView`
  - `VinSnippetView`

- VIN runtime layer
  - `build_vin_snippet_view`, `empty_vin_snippet`
  - `VinOracleBatch`
  - `VinOracleOnlineDatasetConfig`
  - `VinOfflineSourceConfig`
  - `VinDatasetSourceConfig`

- Immutable offline dataset layer
  - `VinOfflineStoreConfig`
  - `VinOfflineWriterConfig`, `VinOfflineWriter`
  - `VinOfflineDatasetConfig`, `VinOfflineDataset`
  - `VinOfflineSample`, `VinOfflineManifest`

- Temporary compatibility exports
  - `OracleRriCache*`
  - `VinSnippetCache*`
  - `repair_*` / `rebuild_*` helpers for the legacy caches

- Migration entry points
  - `scan_legacy_offline_data`
  - `migrate_legacy_offline_data`
  - `verify_migrated_offline_data`

Low-level shard handles, serialization helpers, and migration plumbing remain
internal. The package root intentionally does not export those helpers.

## Internal layout

- `_raw.py`
  - Re-exports the standalone raw ASE/EFM dataset and typed snippet views.

- `_vin_runtime.py`
  - Re-exports the canonical VIN adapter helpers and `VinOracleBatch`.

- `vin_oracle_datasets.py`
  - Split-aware source configs for online VIN data, legacy cache data, and the
    new immutable offline store.

- `_offline_format.py`
  - Manifest, shard, block, and sample-index records for the immutable format.

- `_offline_store.py`
  - Store paths, shard block writers, split-array helpers, and zarr-backed readers.

- `_offline_dataset.py`
  - Runtime reconstruction of `VinOfflineSample` and `VinOracleBatch` from the
    immutable store.

- `_offline_writer.py`
  - Raw-dataset writer for the immutable store plus shard-flush helpers reused
    by migration.

- `_migration.py`
  - Public scan/migrate/verify entry points plus private legacy-conversion plumbing.

- `mesh_cache.py`
  - Standalone processed-mesh cache utilities.

## Immutable offline format

The new on-disk format is:

- `manifest.json`
- `sample_index.jsonl`
- `splits/all.npy`
- `splits/train.npy`
- `splits/val.npy`
- `shards/shard-000000/...`

Each shard stores:

- fixed-size numeric blocks as `zarr` arrays inside the shard group
- optional diagnostic per-row payloads as `msgspec` MessagePack record lists

The training-critical path reads only the fixed blocks:

- VIN points, lengths, and trajectory
- candidate poses and oracle metrics
- PyTorch3D camera tensors
- optional depth blocks
- optional selected backbone tensors

Optional sample-mode diagnostics can additionally decode:

- candidate sampling payloads
- full depth payloads
- candidate point clouds
- full backbone payloads
- future counterfactual payloads

## Migration workflow

Temporary operator tooling lives under:

- `/home/jandu/repos/NBV/.agents/workspace/data_handling_migration/`

The intended workflow is:

1. Scan legacy oracle/VIN caches and emit a migration plan.
2. Convert the legacy data into the immutable VIN offline format.
3. Verify sample counts, split membership, and `(scene_id, snippet_id)` coverage.

The reusable migration logic is implemented in `_migration.py` behind the
public `scan_legacy_offline_data`, `migrate_legacy_offline_data`, and
`verify_migrated_offline_data` entry points. The workspace scripts provide a
thin CLI around those functions for one-off conversion runs.

## Termination criteria for the redesign

The redesign is considered complete when:

- `aria_nbv.data_handling` can be imported without pulling in `aria_nbv.data`.
- the new immutable offline dataset can be written and read end to end,
- Lightning can consume `VinOfflineSourceConfig` for offline training,
- legacy oracle/VIN caches can be migrated into the new format,
- and legacy app/training paths continue to work through compatibility surfaces
  until they are fully switched over.

# `aria_nbv.data_handling`

`aria_nbv.data_handling` is the core data package for Aria-NBV training and
offline diagnostics. It owns:

- the raw ASE/EFM snippet dataset,
- the canonical `EfmSnippetView -> VinSnippetView` adapter,
- the model-facing `VinOracleBatch` contract,
- cache coverage utilities for offline diagnostics,
- the new immutable VIN offline dataset format, and
- migration entry points for legacy oracle and VIN caches.

Code outside this package should import from `aria_nbv.data_handling` rather
than from `aria_nbv.data_handling.*` submodules directly.

## Design goals

- Keep the raw ASE/EFM contract unchanged.
- Have one canonical VIN snippet representation for training.
- Replace per-sample `torch.load` offline datasets with immutable indexed shards.
- Keep training-critical tensors in zarr-backed fixed arrays for fast
  multi-worker random access.
- Keep richer diagnostic payloads lazy and optional.
- Keep the legacy oracle-cache and VIN-snippet-cache readers in one canonical
  package while the immutable store becomes the primary offline path.

## Legacy cutover marker

All runtime code, UI paths, CLIs, and dedicated tests that still belong to the
legacy oracle-cache / VIN-snippet-cache flow are marked with:

- `NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION`

Use this command when the immutable-store migration is complete and the
legacy path should be deleted in one sweep:

```bash
rg -n "NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION" \
  /home/jandu/repos/NBV/aria_nbv/aria_nbv \
  /home/jandu/repos/NBV/aria_nbv/tests
```

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

- Migration entry points
  - `scan_legacy_offline_data`
  - `migrate_legacy_offline_data`
  - `verify_migrated_offline_data`

The package root is now intentionally *canonical-only*. Legacy oracle-cache,
VIN-snippet-cache, and coverage helpers are no longer exported from
`aria_nbv.data_handling` itself. The canonical root also exports
`VinDatasetSourceConfig` instead of the older compatibility alias
`VinOracleDatasetConfig`.

Remaining legacy users should import from the dedicated compatibility modules:

- `_legacy_cache_api.py`
  - grouped legacy cache configs, readers, writers, repair helpers, and
    coverage utilities

- `_legacy_vin_source.py`
  - the temporary `VinOracleCacheDatasetConfig` branch used by the Lightning
    datamodule and diagnostics, plus the backward-compatible
    `VinOracleDatasetConfig` alias

Old direct imports like `aria_nbv.data_handling.oracle_cache` or
`aria_nbv.data_handling.vin_cache` still resolve, but those files are now thin
compatibility wrappers over the real `_legacy_*` owners.

Low-level shard handles, serialization helpers, and migration plumbing remain
internal. The package root intentionally does not export those helpers or the
legacy cache API.

## Internal layout

- `_raw.py`
  - Re-exports the standalone raw ASE/EFM dataset and typed snippet views.

- `_vin_runtime.py`
  - Re-exports the canonical VIN adapter helpers and `VinOracleBatch`.

- `_vin_sources.py`
  - Canonical split-aware source configs for online VIN data and the immutable
    offline store.

- `_legacy_vin_source.py`
  - Dedicated compatibility owner for the legacy cached training source config
    `VinOracleCacheDatasetConfig`.

- `_legacy_cache_api.py`
  - Grouped import surface for the remaining legacy oracle-cache,
    VIN-snippet-cache, and coverage utilities.

- `vin_oracle_datasets.py`
  - Thin compatibility wrapper that preserves the old submodule import path
    while delegating to `_vin_sources.py` and `_legacy_vin_source.py`.

- `oracle_cache.py`, `vin_cache.py`, `vin_provider.py`, `offline_cache_store.py`,
  `offline_cache_serialization.py`, `offline_cache_coverage.py`
  - Thin compatibility wrappers that alias the real legacy implementations in
    the corresponding `_legacy_*` modules.

- `_offline_format.py`
  - Manifest, shard, block, and sample-index records for the immutable format.

- `_offline_store.py`
  - Store paths, shard block writers, split-array helpers, and zarr-backed readers.

- `_offline_dataset.py`
  - Runtime reconstruction of `VinOfflineSample` and `VinOracleBatch` from the
    immutable store.
  - `return_format="vin_batch"` now uses a direct training path that skips
    optional diagnostic payload decoding.

- `_offline_writer.py`
  - Raw-dataset writer for the immutable store plus shard-flush helpers reused
    by migration.

- `_migration.py`
  - Public scan/migrate/verify entry points plus private legacy-conversion plumbing.

- `mesh_cache.py`
  - Standalone processed-mesh cache utilities.

## Immutable offline format

The immutable store is a directory rooted at `VinOfflineStoreConfig.store_dir`.
By default that resolves to `PathConfig().offline_cache_dir / "vin_offline"`,
but migrated or test stores can live anywhere.

The top-level layout is:

- `manifest.json`
- `sample_index.jsonl`
- `splits/all.npy`
- `splits/train.npy`
- `splits/val.npy`
- `shards/shard-000000/...`

Conceptually:

- `manifest.json`
  - Top-level dataset metadata and provenance.
  - Includes `version`, `created_at`, `source`, `oracle`, `vin`,
    `materialized_blocks`, `counterfactuals`, `stats`, `provenance`, and the
    per-shard descriptors in `shards`.
  - The `shards[*].blocks` mapping is the canonical description of what blocks
    exist on disk, their storage kind, dtype, and shape.

- `sample_index.jsonl`
  - One JSON row per globally addressable sample.
  - Rows include `sample_index`, `sample_key`, `scene_id`, `snippet_id`,
    `split`, `shard_id`, and `row`.
  - Migrated stores also preserve legacy provenance fields such as
    `legacy_oracle_key`, `legacy_oracle_path`, `legacy_vin_key`, and
    `legacy_vin_path` when available.
  - `sample_index` is the global row id used by the split arrays.

- `splits/*.npy`
  - `all.npy`, `train.npy`, and `val.npy` are NumPy `int64` arrays of global
    `sample_index` values.
  - These arrays do not duplicate records; they are stable index selections
    into `sample_index.jsonl`.
  - Example: if `val.npy == [3]`, then the fourth row of `sample_index.jsonl`
    is the only validation sample.

- `shards/shard-000000/`
  - One immutable shard directory containing all blocks for a fixed sample row
    range.
  - The shard root has its own `zarr.json`.
  - Numeric tensor blocks are stored as nested Zarr arrays.
  - Optional rich per-row payloads are stored as MessagePack record lists.

Each shard mixes two storage kinds:

- fixed-size numeric blocks as Zarr arrays for fast row-wise random access
- optional diagnostic payload lists as `msgspec` MessagePack files

The path mapping is deterministic:

- logical block name `vin.points_world` becomes Zarr path `vin/points_world/`
- logical block name `oracle.p3d.R` becomes Zarr path `oracle/p3d/R/`
- logical record block `oracle.depths_payload` becomes
  `oracle__depths_payload.msgpack`

Zarr-backed blocks contain:

- a local `zarr.json` metadata file
- chunk payloads under `c/...`

For arrays with a sample-row axis, chunking is row-aligned, so most blocks use
one sample per chunk. That is why a shard often contains many chunk files under
paths such as `oracle/depths/c/...` or `vin/points_world/c/...`.

A cleaned, simplified directory tree looks like:

```text
vin_offline_subset/
├── manifest.json
├── sample_index.jsonl
├── shards/
│   └── shard-000000/
│       ├── zarr.json
│       ├── backbone/
│       │   ├── zarr.json
│       │   ├── cent_pr/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── counts/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── occ_input/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── occ_pr/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pts_world/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── t_world_voxel/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   └── voxel_extent/
│       │       ├── c/
│       │       └── zarr.json
│       ├── backbone__payload.msgpack
│       ├── oracle/
│       │   ├── zarr.json
│       │   ├── candidate_count/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── candidate_indices/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── candidate_poses_world_cam/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── depths/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── depths_valid_mask/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── p3d/
│       │   │   ├── zarr.json
│       │   │   ├── R/
│       │   │   │   ├── c/
│       │   │   │   └── zarr.json
│       │   │   ├── T/
│       │   │   │   ├── c/
│       │   │   │   └── zarr.json
│       │   │   ├── focal_length/
│       │   │   │   ├── c/
│       │   │   │   └── zarr.json
│       │   │   ├── image_size/
│       │   │   │   ├── c/
│       │   │   │   └── zarr.json
│       │   │   ├── in_ndc/
│       │   │   │   └── zarr.json
│       │   │   └── principal_point/
│       │   │       ├── c/
│       │   │       └── zarr.json
│       │   ├── pm_acc_after/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pm_acc_before/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pm_comp_after/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pm_comp_before/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pm_dist_after/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── pm_dist_before/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── reference_pose_world_rig/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   └── rri/
│       │       ├── c/
│       │       └── zarr.json
│       ├── oracle__candidate_pcs.msgpack
│       ├── oracle__candidates.msgpack
│       ├── oracle__depths_payload.msgpack
│       ├── vin/
│       │   ├── zarr.json
│       │   ├── lengths/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   ├── points_world/
│       │   │   ├── c/
│       │   │   └── zarr.json
│       │   └── t_world_rig/
│       │       ├── c/
│       │       └── zarr.json
└── splits/
    ├── all.npy
    ├── train.npy
    └── val.npy
```

Here `c/` contains the actual chunk payload files. Those chunk paths continue
deeper, for example `vin/points_world/c/0/...` or `oracle/depths/c/7/...`, but
the README omits that repeated numeric fan-out for readability.

The training-critical path reads only the numeric Zarr blocks:

- `vin.lengths`, `vin.points_world`, `vin.t_world_rig`
- `oracle.candidate_count`, `oracle.candidate_indices`,
  `oracle.candidate_poses_world_cam`, `oracle.reference_pose_world_rig`,
  `oracle.rri`, and the `oracle.pm_*` metrics
- `oracle.p3d.*` camera tensors
- optionally `oracle.depths` and `oracle.depths_valid_mask`
- optionally selected numeric backbone tensors

Optional sample-mode diagnostics can additionally decode:

- `oracle__candidates.msgpack` for candidate-sampling metadata
- `oracle__depths_payload.msgpack` for full depth payload objects
- `oracle__candidate_pcs.msgpack` for candidate point clouds
- `backbone__payload.msgpack` for full backbone payload objects
- future counterfactual payloads when `materialized_blocks.counterfactuals=true`

Two practical implications:

- The store is intentionally immutable. Rebuilding means writing a new store
  directory rather than editing rows in place.
- Large optional diagnostics can dominate disk usage. In real stores,
  `backbone__payload.msgpack` is often much larger than the fixed Zarr arrays,
  while the Zarr blocks remain the fast path for training and random access.

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

- `aria_nbv.data_handling` can be imported without pulling in deprecated
  `aria_nbv.data` mirror modules.
- the new immutable offline dataset can be written and read end to end,
- Lightning can consume `VinOfflineSourceConfig` for offline training,
- legacy oracle/VIN caches can be migrated into the new format,
- and app/training paths import the canonical `aria_nbv.data_handling` or
  `aria_nbv.utils` owners directly.

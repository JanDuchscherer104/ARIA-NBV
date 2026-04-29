# Data Handling

`aria_nbv.data_handling` owns the current raw-snippet and VIN offline data
contracts. The package root is the public API; internal modules remain private
unless explicitly exported from `__init__.py`.

## Public Surface

- Raw dataset access: `AseEfmDatasetConfig`, `AseEfmDataset`, `EfmSnippetView`,
  `VinSnippetView`, and snippet-loader helpers.
- Oracle batch runtime: `VinOracleBatch`, `VinOracleOnlineDatasetConfig`, and
  `VinDatasetSourceConfig`.
- Immutable offline data: `VinOfflineWriterConfig`, `VinOfflineWriter`,
  `VinOfflineDatasetConfig`, `VinOfflineSourceConfig`,
  `VinOfflineStoreConfig`, `VinOfflineManifest`, and
  `VinOfflineIndexRecord`.
- Offline diagnostics: `collect_vin_offline_dataset_stats` for coverage,
  tensor-shape, RRI, and storage-footprint checks on immutable stores.

The old oracle-cache, VIN-snippet-cache, compatibility wrapper, and migration
modules have been removed. Runtime imports should use the root package exports
or the current private implementation modules directly only inside this
package.

## Immutable VIN Offline Store

The canonical offline format is a strict indexed-shard store:

- `manifest.json` records the store version, source configuration, optional
  materialized block flags, aggregate stats, and shard descriptors.
- `sample_index.jsonl` maps global sample indices to scene/snippet IDs, split
  membership, shard IDs, and shard-local rows.
- `splits/*.npy` stores deterministic train/val/all membership arrays.
- `shards/shard-XXXXXX/` stores fixed numeric blocks as Zarr arrays and optional
  diagnostic payloads as indexed MessagePack record blobs.

`OFFLINE_DATASET_VERSION` is the runtime compatibility gate. When the format
changes, bump the version and rebuild stores with `VinOfflineWriter`; readers
should fail fast on older manifests.

By default `VinOfflineStoreConfig.store_dir` resolves to
`PathConfig().offline_cache_dir / "vin_offline"`. Relative store names such as
`"vin_offline"` are resolved under `offline_cache_dir`.

## Training Source

Lightning consumes offline data through:

```toml
[datamodule_config.source]
kind = "offline"
train_split = "train"
val_split = "val"

[datamodule_config.source.offline]
load_backbone = true
map_location = "cpu"

[datamodule_config.source.offline.store]
store_dir = "vin_offline"
```

`VinOfflineSourceConfig` always returns `VinOracleBatch` samples and disables
diagnostic record loading for the training path. Use `VinOfflineDatasetConfig`
directly when tests or diagnostics need the richer `return_format = "sample"`
path.

## Verification

For data-handling changes, run the tightest relevant checks:

```sh
ruff format aria_nbv/aria_nbv/data_handling/<file>.py
ruff check aria_nbv/aria_nbv/data_handling/<file>.py
uv run pytest tests/data_handling/test_vin_offline_store.py
uv run pytest tests/data_handling/test_public_api_contract.py
```

Broaden to Lightning datamodule tests when source selection or
training-facing batch assembly changes.

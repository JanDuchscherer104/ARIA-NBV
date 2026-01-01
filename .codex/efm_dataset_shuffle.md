# AseEfmDataset shuffling (Dec 2025)

## Problem

`oracle_rri.data.AseEfmDataset` is an `IterableDataset` backed by a WebDataset pipeline created via
`efm3d.dataset.efm_model_adaptor.load_atek_wds_dataset_as_efm`. By default, the resulting sample order is
deterministic (shard order + tar member order), and PyTorch `DataLoader` cannot apply `shuffle=True` for
`IterableDataset`s.

## Solution

Expose the underlying ATEK WebDataset shuffling directly through the EFM3D adaptor API and forward it from
`AseEfmDatasetConfig`:

- Updated `efm3d.dataset.efm_model_adaptor.load_atek_wds_dataset_as_efm(...)` to accept:
  - `shuffle_flag` (bool) → forwarded into `atek.data_loaders.atek_wds_dataloader.load_atek_wds_dataset(..., shuffle_flag=...)`
  - `repeat_flag` (bool) → forwarded into `load_atek_wds_dataset(..., repeat_flag=...)`
- Added `AseEfmDatasetConfig.wds_shuffle` and `AseEfmDatasetConfig.wds_repeat` and pass them into the adaptor.

Implementation:

- `external/efm3d/efm3d/dataset/efm_model_adaptor.py`
- `oracle_rri/oracle_rri/data/efm_dataset.py`

## Usage

For training, enable shuffling on the *train* dataset config:

```py
from oracle_rri.data import AseEfmDatasetConfig

train_ds = AseEfmDatasetConfig(
    scene_ids=["81283"],
    batch_size=None,
    wds_shuffle=True,
    wds_repeat=True,  # typical for training
)
```

Notes:
- `shuffle_flag` uses the ATEK loader’s built-in WebDataset shuffle (buffer size is defined in ATEK; this is not
  currently configurable via our config).
- Use `wds_shuffle=True` only for training; keep validation deterministic.

## Lightning integration

No code changes are required in `oracle_rri/oracle_rri/lightning/lit_datamodule.py`. Shuffling is controlled
entirely through `VinDataModuleConfig.train_dataset` / `val_dataset` (set shuffle only for training).

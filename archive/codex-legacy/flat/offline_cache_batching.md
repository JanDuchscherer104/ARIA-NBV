# Offline cache batching support (OracleRriCacheDataset)

## Summary of changes
- Added `collate_vin_oracle_batches` and helpers in `oracle_rri/oracle_rri/lightning/lit_datamodule.py` to pad variable-length candidate sets, stack `PerspectiveCameras`, and batch `EvlBackboneOutput` fields.
- Relaxed `VinDataModuleConfig` validation to allow `batch_size` for offline-only caches (no appending, no online labeling), with explicit restriction against `include_efm_snippet=True` batching for now.
- Updated DataLoader construction to use the new collate function when offline batching is enabled.
- Updated `VinLightningModule._step` to handle batched VIN outputs without `squeeze(0)`.
- Added unit test `tests/lightning/test_vin_batch_collate.py` for collate padding and backbone stacking.

## Limitations / open issues
- Batched training currently **does not support** `include_efm_snippet=True` (semidense / trajectory features). The collate function raises a clear error in this case.
- OBB-related backbone fields (`obb_pred`, `obbs_pr_nms`, probs lists, etc.) are not batched yet. The collate raises `NotImplementedError` if those fields are present.

## Suggestions / next steps
- If semidense features are required with batch_size > 1, consider caching semidense point tensors in the offline cache (or adding a minimal batchable snippet container) so VIN v2 can consume them without relying on `EfmSnippetView`.
- If OBB features are needed during training, implement a padded/stacked collation strategy for `ObbTW` and list-based probability fields.
- Consider adding a small integration test that loads a real cached sample and runs one forward pass with `batch_size>1` to validate end-to-end behavior.

## Tests run
- `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_vin_batch_collate.py`

(Attempted `uv run pytest` first, but it used the system python without `efm3d`.)

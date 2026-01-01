# Offline Cache Train/Val Split

## Findings
- Train/val split now uses `train_index.jsonl` and `val_index.jsonl`, preserving existing assignments and only filling in new entries as needed.
- `VinDataModule` auto-derives `val_cache` from `train_cache` when `train_val_split` is in (0, 1) and disables `use_train_as_val` to honor the split.
- Offline stats/diagnostics now select the train or val split based on stage and surface split counts in the UI.

## Potential issues
- Changing `train_val_split` after split indices exist does not rebalance; a warning is emitted but existing assignments are kept.
- Split assignment follows index order, so a highly ordered `index.jsonl` may yield biased splits.

## Suggestions
- Add an explicit `split_seed` / `split_strategy` stored in metadata to make splits reproducible and re-creatable.
- Add a CLI/Streamlit action to rebuild split indices when the ratio or dataset changes.
- Consider a lightweight real-data integration test fixture to avoid long-running EVL cache rebuilds.

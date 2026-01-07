# W&B panel patch (2026-01-02)

## Summary of changes
- Added aux metric parsing via `_metric_pairs_with_pattern` in `oracle_rri/configs/wandb_config.py` and wired `wandb.py` to use `train-aux/` + `val-aux/` metrics for calibration and a new VIN aux summary table.
- Updated W&B media browsing to include `*-figures/confusion_matrix*` and `*-figures/label_histogram*` keys (retained legacy keys as fallback).
- Implemented VIN head attribution explorer in `oracle_rri/app/panels/wandb.py`:
  - Loads Lightning checkpoints from `PathConfig.checkpoints`.
  - Loads offline cache batches (`return_format="vin_batch"`) and runs VIN `forward_with_debug`.
  - Attributes predicted RRI score to feature inputs using `AttributionEngine`.
  - Shows group summaries (pose vs global vs extra) and top-k feature contributions.
- Extended `oracle_rri/interpretability/attribution.py` to support vector inputs (feature-level heatmaps) and generalized normalization.
- Added tests:
  - `tests/configs/test_wandb_config.py` for aux metric parsing.
  - `tests/interpretability/test_attribution.py` for vector heatmap normalization.

## Tests run
- `python -m pytest tests/configs/test_wandb_config.py tests/interpretability/test_attribution.py`
- `python -m pytest tests/vin/test_vin_model_v2_integration.py -m integration`

## Notes / suggestions
- Attribution currently targets the VIN head only (features → CORAL logits → expected score). If we want end-to-end attributions (through pose encoding and global pooling), we’ll need to retain the full computation graph instead of using detached `debug.feats`.
- Feature indices are currently unlabeled beyond group buckets; consider emitting named feature metadata from `VinModelV2` to map indices to semantic channels (pose enc dims, global tokens, semidense features) for clearer attribution interpretation.
- Optional enhancement: allow choosing attribution targets (specific CORAL threshold logit vs expected score) and expose Captum noise-tunnel parameters.

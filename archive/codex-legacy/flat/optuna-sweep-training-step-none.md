# Optuna sweep: `training_step` returned `None` (no metrics logged)

Date: 2026-01-06

## Symptom

During `nbv-optuna` sweeps (e.g. `.configs/sweep_config.toml`), PyTorch Lightning emits:

```
/loops/optimization/automatic.py:134: `training_step` returned `None`.
```

and no useful loss/metrics show up in the progress bar / logger.

## Root cause

1. **VIN v2 semidense points were not capped for cached snippets.**
   - `VinModelV2._sample_semidense_points(...)` ignored `max_points` when the snippet payload is a `VinSnippetView`
     (offline cache / vin snippet cache path).
   - Real cached `VinSnippetView.points_world` counts were **huge** (e.g. ~39k–58k points for the first two cache
     entries I inspected), so both semidense projection + (when enabled) PointNeXt processed far more points than
     intended.

2. **PointNeXt does not tolerate NaN padding.**
   - `VinOracleBatch.collate(...)` pads `VinSnippetView.points_world` with `NaN` rows (by design).
   - When Optuna suggests `module_config.vin.use_point_encoder=true`, `VinModelV2._encode_semidense_features(...)`
     passed those padded point tensors into `PointNeXtSEncoder`, which can produce non-finite outputs.

3. **Lightning module skipped the entire batch on *any* non-finite logits.**
   - `VinLightningModule._step(...)` previously did `torch.isfinite(logits).all()` and returned `None` for the whole
     batch if **any** candidate had NaN/Inf logits.
   - With non-finite logits occurring, the training loop effectively produced no loss/metrics.

## Fix implemented

- `oracle_rri/oracle_rri/vin/model_v2.py`
  - `VinModelV2._sample_semidense_points(...)` now truncates `VinSnippetView.points_world` to `max_points` (supports
    both `(N,D)` and `(B,N,D)`).
  - `VinModelV2._encode_semidense_features(...)` now:
    - filters to finite XYZ points per batch element,
    - pads by repeating the last valid point (so the encoder never sees NaNs),
    - uses `torch.nan_to_num(...)` as a final guard,
    - skips samples with no finite points by returning a zero embedding.

- `oracle_rri/oracle_rri/lightning/lit_module.py`
  - `VinLightningModule._step(...)` now **masks non-finite logits per candidate** instead of skipping the whole batch.
  - Logs `"{stage}/drop_nonfinite_logits_frac"` when it drops candidates (outside sanity checking).

## Tests / validation

- Updated masking tests to match the current `VinPrediction` schema and added a regression test for NaN logits masking:
  - `tests/lightning/test_lit_module_masking.py`
- Added unit tests for the new semidense handling logic:
  - `tests/vin/test_vin_v2_semidense_points.py`
- Added + ran an integration test using real ASE data to ensure NaN padding never reaches the (dummy) point encoder:
  - `tests/vin/test_vin_v2_point_encoder_real_data.py`

## Follow-ups / suggestions (not implemented)

- Consider capping semidense points earlier (VIN snippet cache writer / cache dataset config) to avoid large padded
  tensors in collation and reduce CPU→GPU transfer overhead.
- If Optuna pruning relies on step-level validation metrics, ensure the configured `optuna_config.monitor` matches an
  actually logged key (or make the pruning callback robust to `_step`/`_epoch` suffix variants).


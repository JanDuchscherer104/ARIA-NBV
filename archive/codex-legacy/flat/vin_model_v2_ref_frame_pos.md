# VIN v2: reference-frame positional encoding

## What changed
- Positional encoding keys now use `voxel/pts_world` mapped into the **reference rig frame** (not voxel frame).
- Dropped separate voxel-pose encoding from the head to avoid redundant/misaligned context.
- Global attention keys are normalized with an isotropic scale derived from `voxel_extent` and centered using the voxel-grid center (mapped into rig frame).

## Why
- Pose queries are in the reference rig frame; attention keys should be in the same frame for consistent geometry.
- Removing the voxel pose token simplifies the model and reduces frame-mismatch risk.

## Implementation notes
- `_pos_grid_from_pts_world` now takes `pose_world_rig_ref` and computes:
  - `pts_rig = T_rig_world * pts_world`
  - `center_rig = T_rig_world * (T_world_voxel * center_vox)`
  - `pos = (pts_rig - center_rig) / scale` with `scale = 0.5 * max(extent_size)`.
- `head_in_dim` reduced from `pose_dim + pose_dim + field_dim` to `pose_dim + field_dim`.
- Diagnostics no longer expose voxel-pose vectors/encodings.

## Tests run
- `pytest tests/vin/test_vin_model_v2_integration.py` (CPU, real snippet).

## Follow-ups
- If you need axis-aligned normalization in rig frame, consider deriving a tight AABB from `pts_rig` per batch and re-normalizing (slower but exact).
- If attention proves too invariant, consider re-adding a lightweight scalar feature (e.g., `scale` or voxel volume) rather than full voxel-pose encoding.

## Lightning logging batch size
- Updated `VinLightningModule._step` to treat **candidate count** as the effective batch size for logging.
- Passed `batch_size=log_batch_size` to all per-step logs to silence the PL warning about ambiguous batch size inference.
- This does **not** hard-invalidate candidates; it only affects logging aggregation.

## Interval metrics logging
- Added `VinLightningModuleConfig.log_interval_steps` (default 10).
- Metrics (`spearman`, `confusion`, `label_hist`) now update every step and are computed+logged to W&B every `log_interval_steps` steps.
- Reset after each interval to ensure each log covers the last window.
- Epoch-end logging remains for any leftover window.

## Spearman guard
- Guarded `SpearmanCorrCoef.compute()` in both interval and epoch logging to avoid errors when no samples were updated (e.g., sanity check with all-invalid batches).
- Uses the presence of `metric.preds` to decide whether to compute/log.

## Confusion matrix + label histogram logging
- Removed per-entry `log_dict` calls (which produced 200+ scalar metrics).
- Added consolidated visualization logging:
  - Confusion matrix as a heatmap figure.
  - Label histogram as a bar chart figure.
- Logged at interval (`log_interval_steps`) and at epoch end if there is data.
- Uses W&B image logging when `WandbLogger` is active; falls back to `add_figure` for other loggers.
- Guards against empty metrics to avoid noisy warnings or crashes during sanity checks.

## TorchMetrics notes
- TorchMetrics supports plotting for metrics like `MulticlassConfusionMatrix` via `.plot()` (requires visual deps). We use matplotlib directly to avoid extra deps and keep W&B output consistent.

## Sanity-check logging guard
- Added guards to skip interval/epoch metric logging during Lightning sanity checks to avoid W&B step mismatch warnings.
- W&B image logging no longer passes an explicit `step`, preventing out-of-order step warnings.

## Logging safeguards + step alignment
- Skip all logging during Lightning sanity checks to avoid W&B step mismatches.
- Added per-stage `self._metric_has_updates` flag so confusion/hist/spearman are only computed when updated (prevents "compute before update" warnings).
- W&B image logging now uses `step=self.global_step` for monotonic step alignment.

## W&B image logging
- `log_figure` now logs images with `epoch` metadata (no explicit `step`) to match W&B logging style and avoid step ordering warnings.

## lit_module simplification
- Removed per-metric `log_dict` spam and dropped PM distance/acc/comp metrics.
- Restored valid-frac weighting switch (`use_valid_frac_weight`), removed dead/commented code.
- Consolidated scalar logging to a small set: loss, rri_mean, pred_rri_mean, valid_frac_mean, candidate_valid_fraction.
- Interval logging now only applies to TRAIN stage; epoch-end logging remains for all stages.
- Confusion matrix and label histogram now use `_log_figure` (W&B image + epoch), no direct `wandb` import.
- Default `log_interval_steps` set to 10.

## Tests
- `pytest tests/vin/test_vin_model_v2_integration.py` (CPU).

## Metric enum + torchmetrics bundle
- Added `Metric` StrEnum in `oracle_rri/oracle_rri/utils/schemas.py` and exported via `oracle_rri/oracle_rri/utils/__init__.py`.
- Replaced ad-hoc metric strings with enum-driven `_metric_keys()` mapping.
- Introduced `VinMetrics` (torchmetrics.Metric subclass) to encapsulate Spearman + confusion + label histogram and guard empty-state compute.
- Simplified `_step` logging to a single `log_dict` call using enum keys; stage-aware step/epoch logging.
- Interval metrics now use `VinMetrics.compute()` for TRAIN only; epoch metrics log for all stages when available.

## Tests
- `pytest tests/vin/test_vin_model_v2_integration.py` (CPU).

## Metrics package move (rri_metrics)
- Moved logging enums + torchmetrics bundle into `oracle_rri/oracle_rri/rri_metrics/logging.py`.
- Added `VinMetricsConfig` (Config-as-Factory) and wired it into `VinLightningModuleConfig.metrics`.
- Exported `Metric`, `VinMetrics`, and `metric_keys_for_stage` from `oracle_rri/oracle_rri/rri_metrics/__init__.py`.
- Removed `Metric` from `oracle_rri/oracle_rri/utils/schemas.py` + `oracle_rri/oracle_rri/utils/__init__.py`.

## Tests
- `CUDA_VISIBLE_DEVICES="" oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_v2_integration.py`
- Note: system Python 3.12 fails on this test (`coral_pytorch` missing); use the project venv.

## Metric key composition cleanup
- Replaced explicit stage→metric dicts with compositional naming: `metric_key(stage, Metric.X) -> f"{stage}/{suffix}"`.
- `Metric` enum now holds suffixes only (e.g., `loss`, `spearman_step`), reducing duplication.
- Updated `lit_module.py` to pass `stage` into `_log_scalars` and compose keys at log time.
- Updated tests in `oracle_rri/tests/rri_metrics/test_logging_metrics.py`.

## Tests (with Pydantic plugins disabled)
- `PYDANTIC_DISABLE_PLUGINS=1 oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/rri_metrics/test_logging_metrics.py`
- `PYDANTIC_DISABLE_PLUGINS=1 CUDA_VISIBLE_DEVICES="" oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_v2_integration.py`

## Notes
- Without `PYDANTIC_DISABLE_PLUGINS=1`, pytest fails due to an invalid entry point in `.venv`:
  `nbv-summary = oracle_rri.lightning.cli:main --run-mode summarize-vin` (see `oracle_rri/.venv/.../entry_points.txt`).

## Binning + CORAL integration
- Moved ordinal-label→levels conversion into `rri_binning.py` (`ordinal_labels_to_levels`).
- `coral.ordinal_label_to_levels` now delegates to `rri_binning` (logic lives with binning).
- Added `RriOrdinalBinner.labels_to_levels`, `rri_to_levels`, and `expected_from_probs` helpers.
- `lit_module.py` now uses `binner.expected_from_probs(probs)` for predicted RRI proxy.

## File move: rri_metrics
- Moved `coral.py` and `rri_binning.py` into `oracle_rri/oracle_rri/rri_metrics/`.
- Added thin wrappers under `oracle_rri/oracle_rri/vin/` to preserve legacy imports.
- Updated imports in VIN models/pipeline, Lightning module, tests, and `plot_vin_binning.py`.

## Tests
- `PYDANTIC_DISABLE_PLUGINS=1 oracle_rri/.venv/bin/python -m pytest tests/vin/test_coral.py tests/vin/test_rri_binning.py`

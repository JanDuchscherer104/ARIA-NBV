# Grad-norm logging (VinLightningModule)

## Context
- `oracle_rri/oracle_rri/lightning/lit_module.py` now uses a config-driven grad-norm logger instead of a hard-coded list.
- VIN v3 exposes different submodules, so the logger needs to be depth- and pattern-based.

## Implementation summary
- Added `oracle_rri/oracle_rri/utils/grad_norms.py` with:
  - `GradNormLoggingConfig` (`enabled`, `group_depth`, `include`, `exclude`, `norm_type`, `max_items`).
  - `_collect_grad_norm_targets` and `_grad_norm_from_params` helpers.
- `VinLightningModuleConfig.grad_norms` now holds the logging config.
- `on_after_backward` uses the helper to log per-module grad norms.

## Behavior
- Default logs modules at **depth=2** relative to `vin` (e.g., `pose_encoder`, `global_pooler`, `traj_attn`, `semidense_cnn`).
- `include` patterns are **additive** and can log deeper modules like:
  - `global_pooler.pos_grid_encoder`
  - `traj_encoder.pose_encoder.pose_encoder_lff`
- Pattern matching is relative to `vin` and normalizes leading `.` or `vin.` prefixes.
- Grad norms use all parameters in the target module (recurse=True), so parent + child logs overlap by design.

## Tests
- Added `tests/lightning/test_grad_norm_logging.py`:
  - depth-based target selection
  - include-based deep modules
  - norm-type calculation
- Tests run with `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_grad_norm_logging.py` (passed).
- `uv run pytest` still uses system Python here and failed due to missing `power_spherical`.

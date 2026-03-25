# W&B panel comparison update

## Summary
- Added `_select_with_custom` and `_render_run_filters` to reduce UI duplication in `wandb.py`.
- Normalized step bounds via `_normalize_step_bounds` and reused filter values in cache updates.
- Default metric filter updated to `train/coral_loss_rel_random_step|train/pred_rri_mean_epoch|val-aux/spearman|val/coral_loss_rel_random`.
- Applied additional pandas method chaining for the run table and metric curves.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/app/panels/test_wandb_panel.py`
  - Passed.
  - Warnings: deprecation warning from pyparsing and FutureWarnings from PointNeXt AMP decorators during collection.
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_candidate_generation_seed_real_data.py`
  - Passed.
  - Same warnings as above.

## Potential issues / notes
- Step filtering uses summary keys in order: `trainer/global_step`, `global_step`, `_step`, `epoch`. Runs without those keys are excluded when filters are active.
- Entity/project dropdowns rely on the W&B API; if the token lacks permissions, lists may be empty and fall back to custom entry.

## Suggestions
- Consider counting and displaying “runs filtered out due to missing step keys” to clarify step-filter behavior.

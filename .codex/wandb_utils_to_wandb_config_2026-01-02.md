# W&B utils move to wandb_config.py (2026-01-02)

## Changes
- Moved W&B helper utilities from `oracle_rri/oracle_rri/app/panels/wandb_utils.py` into `oracle_rri/oracle_rri/configs/wandb_config.py`.
- Updated `oracle_rri/oracle_rri/app/panels/wandb.py` to import helpers from `oracle_rri/oracle_rri/configs/wandb_config.py`.
- Deleted `oracle_rri/oracle_rri/app/panels/wandb_utils.py` to keep W&B utilities centralized.

## Notes / Findings
- No other in-repo imports of `wandb_utils` remained after the update.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests` failed during collection with
  `ImportError: cannot import name 'backproject_depth' from oracle_rri.rendering.unproject` in
  `tests/rendering/test_unproject.py` (unrelated to W&B changes).

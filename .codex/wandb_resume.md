# W&B resume support

## Summary
- Added `run_id` + `resume` (and `anonymous`) to `WandbConfig`.
- `setup_target()` now forwards `id=run_id` and `resume` into `WandbLogger` (via wandb.init kwargs).

## Files touched
- `oracle_rri/oracle_rri/configs/wandb_config.py`
- `tests/configs/test_wandb_config_resume.py`

## Usage
```toml
[trainer_config.wandb_config]
run_id = "1wz4g6ex"
resume = "allow"  # or "must" / "never"
```

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/configs/test_wandb_config_resume.py`

## Notes
- Lightning's `WandbLogger.log_hyperparams` uses `allow_val_change=True`, so config overwrites are preserved when resuming.

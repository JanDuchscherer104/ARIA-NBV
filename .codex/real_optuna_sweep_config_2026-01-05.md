# Real Optuna Sweep Config (2026-01-05)

## Goal
- Run a real Optuna sweep with validation enabled and `val/loss` as the monitor.
- Train for up to 5 epochs.

## Config file
- `.configs/sweep_config.toml`

## Key settings
- `run_mode = "optuna"`
- `optuna_config.monitor = "val/loss"`
- `trainer_config.enable_validation = true`
- `trainer_config.max_epochs = 5`
- `trainer_config.check_val_every_n_epoch = 1`
- `trainer_config.num_sanity_val_steps = 0`
- `datamodule_config.batch_size = 2`
- `datamodule_config.use_train_as_val = false`
- `datamodule_config.source.train_split = "train"`
- `datamodule_config.source.val_split = "val"`
- `datamodule_config.source.cache.vin_snippet_cache_mode = "required"`
- `datamodule_config.source.cache.vin_snippet_cache_allow_subset = true`

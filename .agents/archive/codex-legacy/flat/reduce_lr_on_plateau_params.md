# ReduceLROnPlateau params (config exposure)

## Summary
- Extended `ReduceLrOnPlateauConfig` to expose core PyTorch scheduler parameters: `mode`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps`.
- `setup_target()` now forwards these into `torch.optim.lr_scheduler.ReduceLROnPlateau`.
- Added a unit test using a lightweight import shim to avoid the `oracle_rri.lightning.__init__` side effects.

## Files touched
- `oracle_rri/oracle_rri/lightning/optimizers.py`
- `tests/lightning/test_reduce_lr_on_plateau_config.py`

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_reduce_lr_on_plateau_config.py`

## Notes
- The test uses a manual module loader to bypass `oracle_rri.lightning.__init__` importing `AriaNBVExperimentConfig` (which currently triggers a pydantic decorator error during collection).

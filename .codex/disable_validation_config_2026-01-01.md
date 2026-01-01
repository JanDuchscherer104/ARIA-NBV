# Disable validation via config (2026-01-01)

## Summary
- Added `enable_validation` flag to `TrainerFactoryConfig` to fully disable validation.
- When disabled, the trainer forces `limit_val_batches=0`, `check_val_every_n_epoch=0`, and `num_sanity_val_steps=0`.

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_trainer_factory.py`

## Tests
- `python -m py_compile oracle_rri/oracle_rri/lightning/lit_trainer_factory.py`

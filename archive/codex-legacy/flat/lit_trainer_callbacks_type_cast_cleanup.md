# lit_trainer_callbacks type-cast cleanup (2025-12-24)

## Changes
- Removed redundant `str`/`int`/`bool` casts in `TrainerCallbacksConfig.setup_target` where config fields are already typed (checkpointing, early stopping, LR monitor, TQDM refresh, backbone finetuning, timer interval).

## Notes / Risks
- Behavior should be identical; relies on Pydantic typing for correct types.

## Tests
- `python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py -q`

## Suggestions
- If more callbacks are added, prefer direct config values over re-casting unless a specific coercion is required.

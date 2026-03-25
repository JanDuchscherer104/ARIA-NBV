# VIN summary torchsummary fix

## Issue
`torchsummary` crashed because it attempted to process a dict input (`efm`) and compared dicts to ints while validating sizes.

## Fix
- Updated `VinLightningModule.summarize_batch` to wrap the VIN model with a dummy-tensor input and capture `efm`/poses/cameras via closure.
- `torchsummary` now receives a simple tensor input and no longer inspects dict sizes.

## Tests
- `ruff format oracle_rri/oracle_rri/lightning/lit_module.py`
- `ruff check oracle_rri/oracle_rri/lightning/lit_module.py`
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_integration.py -m integration`

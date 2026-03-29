# vin_model_v2 summarize_vin uses rich_summary (2025-12-30)

## Summary
- Reworked `VinModelV2.summarize_vin` to build a structured summary dict using `utils.rich_summary.summarize` and render it via `rich_summary` for sanity checks.
- Summary output now includes stats for pose vectors, validity fractions, and outputs while still returning a string for logging/tests.

## Files touched
- `oracle_rri/oracle_rri/vin/model_v2.py`

## Tests
- `python -m pytest tests/vin/test_vin_model_v2_integration.py` (via `oracle_rri/.venv/bin/python`)

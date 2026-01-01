# VIN Streamlit State Rerun Fix

## Issue
- VIN diagnostics state was lost after a Streamlit rerun because the cached
  `VinDiagnosticsState` instance failed `isinstance` checks after code reloads.
  This caused the page to fall back to the "Run the VIN diagnostics" info banner.

## Fix
- In `oracle_rri/app/state.py`, `get_vin_state()` now rehydrates a new
  `VinDiagnosticsState` by copying fields from the previous cached object (or
  dict) instead of discarding it when class identity changes across reloads.

## Tests
- `uv run ruff format oracle_rri/app/state.py`
- `uv run ruff check oracle_rri/app/state.py`
- `uv run pytest tests/test_app_state_signature.py`

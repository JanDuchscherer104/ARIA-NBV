# VIN v3 split cleanup (2026-01-07)

## What changed
- Mainline `oracle_rri/vin` now only exposes VIN-Core (v3) components; v1/v2 + plotting/diagnostics live under `oracle_rri/vin/experimental`.
- Added experimental support files: `vin/experimental/types.py` (VinForwardDiagnostics/VinV2ForwardDiagnostics + re-exports), `vin/experimental/pose_encoding.py` (FourierFeatures + LFF re-exports).
- Restored `vin/experimental/plotting.py` and fixed its relative imports for the new package depth.
- Fixed relative imports in `vin/experimental/model.py` and `vin/experimental/model_v1_SH.py` to point at top-level modules after the move.
- Updated app + lightning imports to use experimental plotting/diagnostics and new model paths.
- Updated tests that referenced v1/v2 paths and old Lightning/DataModule config signatures to match the v3-only mainline + experimental split.
- Updated doc source references in `.qmd` files to point to `vin/experimental/model*.py`.

## Test results (real data)
- `oracle_rri/tests/vin/test_types.py`
- `oracle_rri/tests/vin/test_vin_model_v3_core.py`
- `oracle_rri/tests/integration/test_vin_v3_real_data.py`
- `oracle_rri/tests/integration/test_vin_lightning_real_data.py`

All passed via:
`oracle_rri/.venv/bin/python -m pytest ...`

## Known issues / follow-ups
- `ruff check oracle_rri/oracle_rri/vin/experimental/model_v1_SH.py` raises `N999 Invalid module name` because of the filename. I ran `ruff check --ignore N999` for this file; consider a per-file ignore in config if you want full lint green without renaming.
- Generated docs (`docs/*.html`, `docs/search.json`) are now stale relative to updated `.qmd` file paths; re-run Quarto to regenerate when convenient.

## Suggestions
- If you plan to keep VIN v1_SH long-term, consider renaming `model_v1_SH.py` to a lint-friendly name and updating imports.
- Optionally expose more `vin.experimental` symbols in `vin/experimental/__init__.py` for convenience.

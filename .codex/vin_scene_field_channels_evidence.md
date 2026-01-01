# VIN Scene Field Channels Evidence

## Change
- Added `build_scene_field_evidence_figures(...)` in `oracle_rri/vin/plotting.py` to visualize `debug.field_in` channels (ordered by `scene_field_channels`) as world-space scatters.
- Backbone Evidence tab now renders these channels first (falling back to `build_backbone_evidence_figures` if needed).

## Tests
- `uv run ruff format oracle_rri/vin/plotting.py oracle_rri/vin/__init__.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run ruff check oracle_rri/vin/plotting.py oracle_rri/vin/__init__.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run pytest ../tests/vin/test_vin_plotting.py`

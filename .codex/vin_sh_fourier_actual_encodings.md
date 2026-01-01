# VIN SH/Fourier Actual Encodings

## Change
- Added `build_candidate_encoding_figures(...)` in `oracle_rri/vin/plotting.py` to visualize actual candidate SH components, radius Fourier features, and pose encoder embeddings as heatmaps.
- Wired the SH/Fourier Streamlit tab to render these actual encodings alongside the conceptual bases and to save them to HTML when requested.

## Tests
- `uv run ruff format oracle_rri/vin/plotting.py oracle_rri/vin/__init__.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run ruff check oracle_rri/vin/plotting.py oracle_rri/vin/__init__.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run pytest ../tests/vin/test_vin_plotting.py`

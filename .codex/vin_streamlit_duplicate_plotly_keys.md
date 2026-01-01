# VIN Streamlit Plotly Keys

## Issue
- VIN diagnostics tabs rendered multiple Plotly figures with identical parameters,
  triggering Streamlit `StreamlitDuplicateElementId` errors.

## Fix
- Added stable `key=` values to `st.plotly_chart(...)` calls within loops in
  `oracle_rri/app/panels.py` (alignment, field slices, field/token histograms,
  backbone evidence).

## Tests
- `uv run ruff format oracle_rri/app/panels.py`
- `uv run ruff check oracle_rri/app/panels.py`

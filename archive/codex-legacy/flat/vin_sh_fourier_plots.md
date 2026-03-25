# VIN SH/Fourier Encoding Plots (Docs)

## What changed
- Added an "Encoding visualizations" block to `docs/contents/impl/vin_nbv.qmd` that embeds the actual SH component heatmaps and radius Fourier feature plots from `docs/figures/impl/vin/`.

## Why
- The SH/Fourier section explained the math but did not show the actual encoding outputs. The added figures make the basis and radius encodings concrete.

## Notes / Follow-ups
- If you want the figures to stay in sync with future config changes, consider adding a small docs-generation step that calls `oracle_rri.vin.plotting.build_vin_encoding_figures` and writes the PNGs into `docs/figures/impl/vin/`.

## Tests
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_vin_real_data.py -q`

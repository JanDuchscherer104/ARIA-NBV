# RRI Binning panel (2026-01-01)

## Summary
- Added a dedicated Streamlit page for RRI binning diagnostics.
- Page loads only `rri_binner_fit_data.pt` + `rri_binner.json` and renders:
  - Plotly histogram of raw RRIs with quantile edge lines.
  - Plotly label histogram derived from edges.
  - Random-guess CORAL loss baseline `(K-1)*log(2)` and basic RRI stats.

## Files touched
- `oracle_rri/oracle_rri/app/panels.py`
- `oracle_rri/oracle_rri/app/app.py`

## Notes / suggestions
- If the label histogram is far from uniform, refit the binner on a larger, representative offline sample set.
- Consider adding a quick “uniformity score” (e.g., KL divergence from uniform) if you want a numeric alert.

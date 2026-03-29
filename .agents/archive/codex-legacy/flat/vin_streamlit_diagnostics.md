# VIN Streamlit Diagnostics (2025-12-29)

## Summary of changes
- Added a new Streamlit page `VIN Diagnostics` that runs VIN `forward_with_debug` on oracle batches using `AriaNBVExperimentConfig` and visualizes internal tensors via Plotly/matplotlib.
- Implemented independent session cache for VIN diagnostics (`VinDiagnosticsState`) so the page does not rely on the existing `PipelineController`/`AppState` flow.
- Wired the new page into app navigation and documented it in `docs/contents/impl/data_pipeline_overview.qmd`.

## Key files touched
- `oracle_rri/oracle_rri/app/panels.py`: new VIN diagnostics UI, state cache, plotting helpers.
- `oracle_rri/oracle_rri/app/app.py`: added `VIN Diagnostics` page to navigation.
- `docs/contents/impl/data_pipeline_overview.qmd`: updated list of Streamlit pages.

## Potential issues / risks
- VIN diagnostics does heavy compute inside the panel (labeler + EVL forward). This diverges from the original “no heavy compute in panels” guideline but is required for independent diagnostics.
- The diagnostics page currently caches module/datamodule in `st.session_state`; if the EVL cfg/ckpt or dataset paths are invalid, errors are surfaced but not logged with stack traces.
- `plot_vin_encodings_from_debug` writes files under `.logs/vin/streamlit`; ensure this is acceptable for your artifact policy.

## Suggestions / next steps
- Optionally add a TOML picker UI (file browser) or allow overriding datamodule/labeler parameters (e.g., `max_candidates`) to make the diagnostic runs faster.
- Consider adding a “fetch snippet” option if you want mesh/trajectory overlays in the VIN diagnostics page (requires `EfmSnippetView`).
- If heavy compute inside panels is a concern, wrap the VIN diagnostics compute path in a lightweight controller-like helper and keep the panel focused on UI.

## Tests
- `pytest tests/vin/test_vin_diagnostics.py` failed during collection: `ModuleNotFoundError: No module named 'coral_pytorch'`.
  - This dependency needs to be installed in the active environment to run VIN integration tests.

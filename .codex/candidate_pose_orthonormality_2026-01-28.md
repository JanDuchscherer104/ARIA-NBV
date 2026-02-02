# Candidate Pose Orthonormality Diagnostics (2026-01-28)

## Summary
- Added orthonormality metrics to the Streamlit Candidate Poses page for `reference_pose` and `sampling_pose`.
- Added unit tests for the new orthonormality stats helper.
- Updated Streamlit usage docs to mention the orthonormality expander.

## Findings
- Plotting chain uses `PoseTW.R` directly (optional `rotate_yaw_cw90` in `SnippetPlotBuilder.add_frame_axes`).
- The plotting stack does not renormalize axes; any non-orthonormality likely comes from the input poses, not the renderer.

## Changes
- `oracle_rri/oracle_rri/app/panels/candidates.py`: add `_pose_orthonormality_stats` + `_render_pose_orthonormality` and UI expander.
- `tests/app/panels/test_candidates_panel.py`: new tests for orthonormality metrics.
- `docs/contents/impl/data_pipeline_overview.qmd`: note new orthonormality expander.
- `.codex/AGENTS_INTERNAL_DB.md`: record `uv run pytest` env gotcha.

## Tests
- `ruff format oracle_rri/oracle_rri/app/panels/candidates.py tests/app/panels/test_candidates_panel.py`
- `ruff check oracle_rri/oracle_rri/app/panels/candidates.py tests/app/panels/test_candidates_panel.py`
- `oracle_rri/.venv/bin/python -m pytest tests/app/panels/test_candidates_panel.py`
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`
- `quarto render docs/contents/impl/data_pipeline_overview.qmd --to html`

## Follow-ups
- If orthonormality metrics show significant error, trace upstream pose construction (candidate sampling or cached pose serialization) rather than plotting.

## Update
- Switched candidate plots to orthographic projection (with a UI toggle) to avoid perspective skew that makes orthonormal axes appear non-orthogonal.
- Added Plotly projection helper + test.

## Tests (update)
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/pose_generation/test_plotting_helpers.py tests/app/panels/test_candidates_panel.py`
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Update (view direction vectors)
- Added optional view-direction vectors to the rig-frame positions plot via a checkbox.
- Added plotting helper support (`plot_position_sphere(..., dirs=...)`) and tests.

## Tests (view dirs)
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/pose_generation/test_plotting_helpers.py`
- `quarto render docs/contents/impl/data_pipeline_overview.qmd --to html`
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Update (shell directions)
- Compute shell offsets + directions in reference frame for plotting.
- Direction plots now default to shell directions (fallback to valid), preventing empty sphere plot.
- Position overlay uses shell directions masked by `mask_valid` to align with plotted points.
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

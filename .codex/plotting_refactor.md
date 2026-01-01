# 2025-11-24 – Plotting refactor & viz utils

- Added reusable `add_points`, `add_candidate_frusta`, and `add_sampling_shell` methods to `SnippetPlotBuilder` so 3D plots share a common builder API and auto-expand scene bounds.
- Refactored `pose_generation.plotting` and `streamlit_app` to consume the builder instead of bespoke Plotly wiring; candidate/rejected plots now reuse the same logic and shell plots finally display sampled points.
- Introduced `oracle_rri/utils/viz_utils.py` as the home for previously shared free functions (re-exported from `data.utils` for compatibility).
- Tests run: `pytest oracle_rri/tests/test_plotting_frustum_device.py oracle_rri/tests/data_handling/test_utils.py` (all green, only torch jit deprecation warnings).

Open follow-ups / notes
- Consider adding targeted tests for the new builder methods (candidate frusta/sampling shell) and ensuring scene range padding remains adequate for far candidates.
- Streamlit frustum plot still uses the first RGB calibration for all candidates; revisit if per-frame intrinsics are required.

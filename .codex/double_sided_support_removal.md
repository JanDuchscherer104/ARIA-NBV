# Double-sided & Proxy Wall Options Removal (2025-11-24)

## What changed
- Removed double-sided mesh rendering and proxy wall injection from `oracle_rri/data/plotting.py` to simplify Plotly mesh traces.
- Streamlit UI no longer exposes the "double-sided walls" layer toggle or CPU renderer `add_proxy_walls` knob; defaults from configs remain.
- Updated `pose_generation/plotting.py` to match the simplified mesh helper signature.
- Dropped `_aligned_pose_world_cam`; frustum poses now slice trajectory/camera by frame index directly inside `plot_trajectory`.
- Refactored `SnippetPlotBuilder` to store the snippet once (`from_snippet`) and accept only visual customisation params; `streamlit_app` and `plot_trajectory` now use this pattern. Candidate plotting no longer depends on `SnippetPlotBuilder`.
- Removed Streamlit console/log panel and interactive Python console; app no longer collects or displays inline logs.
- Streamlit candidate frustums now use Plotly frustum helpers from `data.plotting` (no `pose_for_display`).
- Removed `plot_trajectory`; Streamlit now builds trajectory/mesh/frusta plots directly with `SnippetPlotBuilder`.

## Rationale
- Requested to remove support for `double_sided` and `add_proxy_walls` in plotting and Streamlit. Mesh visuals now use a single outward-facing Mesh3d trace; proxy walls are not auto-added.

## Implications / follow-ups
- Depth renderers still support `add_proxy_walls`; UI removal means they always use config defaults. If users need to toggle this again, it must be reintroduced deliberately.
- Any plots relying on interior visibility may look dimmer when the camera is inside meshes. Consider adding per-scene opacity tweaks if needed.
- Other files had pre-existing local changes; none were touched.

## Tests
- `oracle_rri/.venv/bin/pytest oracle_rri/tests/test_plotting_semidense.py`

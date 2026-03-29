# Depth hit backprojection unification

- Streamlit depth page now reuses `build_candidate_pointclouds` for backprojection, eliminating the duplicate path in `RenderingPlotBuilder.add_depth_hits`.
- UI keeps stride selection and adds a max-points cap; selected candidates are backprojected once via the shared batch routine and visualized with frusta.
- Cached point clouds per depth batch + stride in session state (`nbv_candidate_pcs`) to avoid recomputation across depth/rri pages; cache invalidated whenever data/candidates/depth rerun in `app.py`.
- Impact: depth renders shown in the dashboard match the point clouds used for RRI scoring, reducing divergence between debug plots and metrics.
- Follow-up: consider deprecating or refactoring `add_depth_hits` to delegate internally to `build_candidate_pointclouds` to avoid future drift.

# Render RRI page simplification (2025-12-08)

- The empty "Oracle RRI per candidate" chart traced to a typo in `OracleRRI.score`: it computed `(cd_before - cd_before) / cd_before`, forcing RRI to zero. Added a shared `_score_single` with a safe `(before - after) / before` calculation and a batch helper that reuses the cropped mesh.
- Simplified `render_rri_page`: moved RRI scoring into `_compute_oracle_rri`, dropped the verbose occupancy table, unified candidate selection for charts/3D view, and reused the expanded occupancy bounds from back-projected candidates.
- New `score_batch` path avoids repeated mesh crops and keeps the dashboard code leaner while preserving per-candidate diagnostics.
- Testing notes: `pytest oracle_rri/oracle_rri/dashboard/panels.py` fails when collected as a test module (relative import / streamlit dependency). Use package-aware tests or mark skips; ensure `streamlit` is available in the test environment.
- RRI metric update (batch-aware): added `chamfer_point_mesh_batched` + per-example variant of `point_mesh_face_distance` so candidates sharing the same GT mesh can be scored in one call while keeping per-candidate losses. `OracleRRI.score_batch` now uses this vectorised path.
- RRI metric update v2: `chamfer_point_mesh_batched` is fully vectorised (no Python loops) and returns accuracy/completeness/bidirectional per candidate. `chamfer_point_mesh` now wraps the same path. `RriResult` carries accuracy/completeness before/after, and the dashboard plots point→mesh and mesh→point bars alongside RRI and Chamfer.

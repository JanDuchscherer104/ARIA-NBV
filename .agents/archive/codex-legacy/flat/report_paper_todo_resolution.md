# Paper TODO Resolution Report (2026-01-29)

## Scope
- Resolved all dashy-todo markers in `docs/typst/paper/**` by aligning text with code/config.
- Left only plain `// TODO` comments for items that still need future data.

## Sources Used
- Code: `oracle_rri/oracle_rri/lightning/lit_module.py`, `oracle_rri/oracle_rri/vin/model_v3.py`,
  `oracle_rri/oracle_rri/rendering/candidate_pointclouds.py`,
  `oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py`,
  `oracle_rri/oracle_rri/data/efm_views.py`,
  `oracle_rri/oracle_rri/data/vin_snippet_utils.py`,
  `oracle_rri/oracle_rri/pose_generation/candidate_generation.py`,
  `oracle_rri/oracle_rri/pose_generation/candidate_generation_rules.py`
- Config/data: `.configs/offline_only.toml`, `.configs/paper_figures_oracle_labeler.toml`,
  `docs/typst/slides/data/offline_cache_stats.json`
- Docs: `docs/typst/paper/**`, `docs/typst/shared/macros.typ`, `docs/index.qmd`,
  `docs/contents/todos.qmd`

## Key Findings Applied
- Pytorch3D rendering is metric depth (`PerspectiveCameras` with `in_ndc=false`); backprojection
  uses pixel centers converted to PyTorch3D NDC before `unproject_points`.
- Candidate pose sampling applies `rotate_yaw_cw90` to match EFM3D convention.
- EVL voxel extent defaults to `[-2, 2, 0, 4, -2, 2]`.
- Offline cache stats: 80 scenes, 883 snippets (706 train / 177 val), 4608 snippets total.
- VINv3 baseline config (field_dim 24, head_hidden_dim 192, head_num_layers 2, etc.) and
  coverage weighting schedule (strength 0.6 → 0 across 6 epochs).

## Remaining TODOs (as comments)
- Runtime profiling + memory table pending actual measurements.
- Optuna appendix needs CSV + run IDs for plots.

## Output Verification
- `typst compile docs/typst/paper/main.typ /tmp/paper.pdf --root docs` succeeded.

## Suggestions
- Add runtime profiling run to replace remaining TODO comment.
- Export Optuna CSV and update appendix plots to remove placeholder comment.

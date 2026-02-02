# paper_figures_oracle_labeler.toml sync

- Updated `.configs/paper_figures_oracle_labeler.toml` to match Python defaults:
  - generator: num_samples=60, oversample_factor=2.0, min_radius=0.5, max_radius=1.8, min_elev_deg=-20, min_distance_to_mesh=0.2, ray_subsample=32, step_clearance=0.1, device=auto.
  - depth: max_candidates_final=60, device=auto; renderer cull_backfaces=true.
- Copied the config into `docs/typst/slides/data/paper_figures_oracle_labeler.toml` to satisfy Typst root restrictions.
- Slides and paper tables now load values from the TOML copy via `toml()`:
  - `docs/typst/slides/slides_4.typ`
  - `docs/typst/paper/sections/08-system-pipeline.typ`

Note: Typst compile root is `docs`, so configs outside the root must be mirrored inside `docs/`.

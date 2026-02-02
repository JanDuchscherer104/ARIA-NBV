# Paper: Slide-Figure Gallery Removal (12i) — Migration Notes

## Goal
Remove `docs/typst/paper/sections/12i-appendix-slide-figures.typ` and place its
figures next to the relevant narrative sections (pipeline, oracle, VIN, wandb).

## What changed
- Deleted `docs/typst/paper/sections/12i-appendix-slide-figures.typ` and removed
  its include from `docs/typst/paper/main.typ`.
- Re-homed all unique figures from 12i into their “natural” sections (below).
- Updated text references that previously pointed to the removed appendix
  section (`@sec:appendix-slide-figures`).

## Figure mapping (old 12i → new location)

### Pipeline / UI
- `/figures/app/traj.png` + `/figures/app/semidense.png`
  - Now in `docs/typst/paper/sections/08-system-pipeline.typ` as `@fig:streamlit-diagnostics`

### Candidate sampling
- `/figures/app-paper/pos_ref.png` + `/figures/app-paper/view_dirs_ref.png` + `/figures/app-paper/orientation_jitter.png`
  - Now in `docs/typst/paper/sections/08-system-pipeline.typ` as `@fig:candidate-poses`
  - Fixed a caption mismatch (previously claimed renders/histograms).

### Oracle rendering / backprojection diagnostics
- `/figures/app-paper/cand_renders_1x3.png` + `/figures/app-paper/depth_histograms_3x3.png`
  - Now in `docs/typst/paper/sections/05-oracle-rri.typ` as `@fig:oracle-depth-diagnostics`
- `/figures/app-paper/backproj+semi.png` + `/figures/app-paper/semi-dense-pc-cand-vis.png`
  - Now in `docs/typst/paper/sections/05-oracle-rri.typ` as `@fig:oracle-fusion-diagnostics`

### VINv3 architecture diagrams / semidense diagnostics
- `/figures/diagrams/vin_nbv/mermaid/head.png` + `/figures/diagrams/vin_nbv/mermaid/global_pool.png`
  + `/figures/diagrams/vin_nbv/mermaid/pose_encoder.png` + `/figures/diagrams/vin_nbv/mermaid/semidense_proj.png`
  - Now in `docs/typst/paper/sections/06-architecture.typ` as `@fig:vin-arch-diagrams`
- `/figures/diagrams/vin_nbv/mermaid/semidense_frustum.png`
  - Now in `docs/typst/paper/sections/06-architecture.typ` as `@fig:vin-semidense-frustum`
- `/figures/app-paper/semi-dense-counts-proj.png` + `/figures/app-paper/semi-dense-weight-proj.png`
  + `/figures/app-paper/semi-dense-std-proj.png`
  - Now in `docs/typst/paper/sections/06-architecture.typ` as `@fig:app-paper-semidense-proj`

### Duplicates that were already present elsewhere
- VIN input superposition (`/figures/app-paper/vin-geom-oc_pr-candfrusta-semi-dense.png`)
  - Already in `docs/typst/paper/sections/06-architecture.typ` as `@fig:vin-inputs`
- EVL field slices (`field_occ_in`, `field_occ_pr`, `field_counts_norm`)
  - Already in `docs/typst/paper/sections/06-architecture.typ` as `@fig:evl-field-slices`
- CORAL binning figures
  - Already documented in `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`
    (`@fig:coral-binning-overview`, `@fig:coral-binning-stats`), so the 12i duplicates were removed.

### Training curves (W&B)
- `/figures/wandb/*`
  - Now in `docs/typst/paper/sections/09c-wandb.typ` as `@fig:wandb-coral` and `@fig:wandb-aux`
  - `docs/typst/paper/main.typ` now includes `sections/09c-wandb.typ`.

## Reference updates (removed “slide gallery appendix”)
- `docs/typst/paper/sections/01-introduction.typ`: now points to `@sec:wandb-analysis` + `@sec:appendix-extra`
- `docs/typst/paper/sections/05-oracle-rri.typ`: now points to the new oracle diagnostics figures
- `docs/typst/paper/sections/08a-frustum-pooling.typ`: now references `@fig:oracle-fusion-diagnostics`
- `docs/typst/paper/sections/09-diagnostics.typ`: now references pipeline/oracle figures directly
- `docs/typst/paper/sections/09c-wandb.typ`: curves are embedded (no appendix indirection)
- `docs/typst/paper/sections/12b-appendix-extra.typ` + `docs/typst/paper/sections/12-appendix-gallery.typ`: updated to avoid `@sec:appendix-slide-figures`

## “Unclear figure” checks
- `08-system-pipeline.typ`: fixed a caption that described images that were not shown.
- `06-architecture.typ`: removed stale TODO marker and added a short note explaining why
  `counts_norm` slices can look “odd” (log-scaled, per-snippet normalization).

## Follow-ups (optional)
- Re-run a paper build to verify cross-references and figure placement.
  (In this environment, `typst` via snap is sandbox-blocked.)

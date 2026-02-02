# Semidense slides + paper consistency update (2026-01-29)

## Scope
- Align semidense notation and reliability-weight formulas in slides.
- Integrate semidense projection plots and clean slide layout.
- Cross-check paper/macros for inconsistencies.

## Key findings
- `inv_dist_std` in Aria/ASE is **sigma_rho** (std of inverse distance), not `1/sigma_d`.
- Semidense projection grids are built by **binning screen-space pixel coords** into a coarse `G_sem x G_sem` grid (G=12 for current configs), regardless of image size (H=W=120). Each bin aggregates counts and depth stats.

## Changes made
- `docs/typst/slides/slides_4.typ`:
  - clarified `inv_dist_std` as `sigma_rho` on semidense inputs,
  - fixed `G_sem` notation and `F_cnn` shape symbol,
  - added reliability-weight formulas + validity mask,
  - redesigned “Semidense projection: transforms + bins” slide (clean, no overflow),
  - added “Semidense projection maps” slide with the three PNGs.
## Checked (no edits needed)
- `docs/typst/shared/macros.typ`: already defines `symb.shape.Fcnn` and `symb.shape.Gsem`.
- `docs/typst/paper/sections/*`: semidense sections already use `sigma_(rho)` for `inv_dist_std`.

## Render outputs
- PNGs for inspection: `.codex/semidense_slides_png_v2/050.png`–`055.png`, `063.png`.

## Open items / suggestions
- If desired, fix remaining **code comments** in `oracle_rri/oracle_rri/vin/model_v3.py` that still describe `inv_dist_std` as `1/sigma_d` (docs now use sigma_rho).
- `tokens.py` UI text still labels `inv_dist_std` as `1/σ_d`; update if you want the diagnostics popover to match the paper/slides.

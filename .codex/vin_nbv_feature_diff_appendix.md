# VIN-NBV Feature Diff Appendix (2026-01-26)

## Scope
Added a new appendix subsection comparing VIN-NBV (Frahm 2025) features with VINv3, plus actionable modifications to `vin/model_v3.py`.

## Sources
- `literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex` (VIN-NBV design).
- `oracle_rri/oracle_rri/vin/model_v3.py` (current v3 features).

## Summary of Added Content
- VIN-NBV feature pipeline (normals + visibility + depth, projection grid, variance, emptiness, base-view count, CNN encoder).
- VINv3 pipeline (EVL voxel field + pose encoding + semidense projection stats + FiLM + CORAL head).
- Key differences and missing signals.
- Actionable suggestions for v3: stage cue, emptiness proxy, variance cue, optional tiny 2D encoder.

## Files Updated
- `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ`

## Build Check
- `typst compile --root docs docs/typst/paper/main.typ`

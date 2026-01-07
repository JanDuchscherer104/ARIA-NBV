# VIN v2 paper update (2026-01-06)

## Objective
Bring the Typst paper (`docs/typst/paper/main.typ`) up-to-date with the *current* VIN v2 architecture in
`oracle_rri/oracle_rri/vin/model_v2.py`, and ensure the document compiles locally.

## What changed (paper)
- Updated abstract + intro to reflect VIN v2 modules: EVL backbone, pose-conditioned voxel attention, semidense projection stats, semidense frustum MHCA, optional PointNeXt, trajectory encoder, and voxel-valid-fraction gating.
- Reworked architecture section to match implementation concepts and naming (incl. `semidense_candidate_vis_frac` and `voxel_valid_frac` gating/feature).
- Updated frustum pooling section to define token features `(u, v, z, inv_dist_std, obs_count)` and to document visibility embedding + optional masking.
- Updated training config snapshot + ablation plan to list the *actual* toggles used in sweeps / experiments.
- Replaced multiple outdated figures with placeholders to be filled with Streamlit exports (VIN diagnostics + training curves).

## What changed (bibliography)
- Added arXiv references:
  - Transformer attention (`Transformer-vaswani2017`)
  - GroupNorm (`GroupNorm-wu2018`)
  - OneCycle learning-rate policy (`OneCycleLR-smith2018`)
- Synced bibliography for Typst paper compilation: `docs/references.bib` → `docs/_shared/references.bib`.

## What changed (code terminology for consistency)
- Renamed semidense projection “valid fraction” feature to `semidense_candidate_vis_frac` and kept backward-compatibility aliases (legacy keys still logged/available).
- Updated the VIN logging + tests accordingly.

## How to compile (verified)
- `mkdir -p .codex/_render docs/_shared && cp -f docs/references.bib docs/_shared/references.bib`
- `typst compile --root docs docs/typst/paper/main.typ .codex/_render/paper.pdf --diagnostic-format short`
- Optional preview:
  - `typst compile --root docs -f png --ppi 200 --pages 1-2 docs/typst/paper/main.typ .codex/_render/paper-{0p}.png`

## Notes / TODOs
- Replace placeholder figure blocks with Streamlit exports (VIN v2 architecture, training curves, and any other outdated figures you want to drop).
- Consider updating `docs/typst/paper/sections/09c-wandb.typ` to cover the *Optuna sweep* run set (current content is a Jan 3 snapshot).


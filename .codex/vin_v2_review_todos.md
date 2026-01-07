# VIN v2 review todos (added)

## Context
User requested adding all todos from the architecture review plus additional findings. Semidense projection is confirmed as part of the official architecture; FiLM usage remains undecided.

## Added to `docs/contents/todos.qmd`
- Align `VinPrediction` dataclass fields with call sites (`candidate_valid`, `valid_frac`).
- Validate `scene_field_channels` against the field aux map or implement missing channels (`observed`, `free_input`, `new_surface_prior`).
- Fix `counts_norm` clamp to enforce `[0,1]` on the ratio.
- Fix K/V separation in `PoseConditionedGlobalPool` so positional encoding affects keys only.
- Update docs to reflect semidense projection + trajectory context + `apply_cw90_correction` semantics + actual channel defaults.
- Add tests for `counts_norm` range, channel compatibility, and K/V positional encoding behavior.

## Updates (2026-01-02)
- Implemented derived scene-field channels (`observed`, `unknown`, `free_input`, `new_surface_prior`) with explicit validation.
- Fixed `counts_norm` clamp to enforce `[0,1]` after log normalization.
- Fixed PoseConditionedGlobalPool K/V separation and added LayerNorm + residual + MLP block.
- Added semidense projection features + semidense frustum MHCA summary, and wired `valid_frac`/`candidate_valid` in v2.
- Updated `docs/contents/impl/vin_nbv.qmd` to reflect current v2 architecture (semidense projection + MHCA, defaults, etc).
- Added unit tests + real-data integration coverage for the changes.

## Open decision
- Whether to keep FiLM or remove it (now documented; currently kept as a lightweight conditioning option).

## Remaining follow-ups
- Candidate-relative positional encoding for global attention keys.
- Stage-aware features or binning (VIN-NBV stage normalization).
- Learnable channel gates/thresholds for noisy fields (e.g., `cent_pr`).
- Learnable CORAL bin shifts.
- Optional RGB/DINOv2 candidate-plane features.

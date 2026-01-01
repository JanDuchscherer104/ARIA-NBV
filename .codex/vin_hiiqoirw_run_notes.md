# VIN run hiiqoirw - training dynamics and follow-ups

## Context
- W&B run: `traenslenzor/aria-nbv/hiiqoirw` (name `R2025-12-29_18-12-16`, created `2025-12-29T17:12:18Z`).
- Config highlights: LFF-only pose encoding (no SH), `scene_field_channels=["occ_pr"]`, `global_pool_mode=attn`, `use_unknown_token=True`, `use_valid_frac_feature=True`, `candidate_min_valid_frac=0.2`.

## Key training dynamics (161 logged steps, up to step 329)
- Loss stays near CORAL baseline: mean **9.37** (min 7.05, max 13.64), only slight change from early to late (9.23 -> 9.18).
- `pred_rri_mean_step` nearly constant **0.605-0.633** (std ~0.008); correlation with `rri_mean_step` ~**0.0** -> score collapse / underfitting.
- `voxel_valid_fraction_step` mean **0.38** (min 0.09, max 0.59) -> many frustum samples outside voxel grid; local features dominated by unknown token.
- `candidate_valid_fraction_step` mean **0.71** (min 0.19, max 1.0) -> ~29% candidates excluded by validity mask.
- Loss vs `rri_mean_step` correlation **-0.46** -> model does relatively better on high-RRI batches but struggles on low-RRI discrimination.

## Likely causes (architecture-linked)
- Occ_pr-only scene field removes counts/unknown/new-surface cues, limiting the model to occupancy probability and weakening RRI-aligned signals.
- LFF-only pose encoding reduces directional inductive bias (no SH), increasing sample complexity and slowing learning in short runs.
- Low frustum validity (0.38 mean) + valid-frac weighting reduces gradient magnitude, especially when local tokens are mostly unknown.
- Frustum depths `[0.5, 1, 2, 3]` likely too shallow for large-room geometry; FOV clamp may under-sample wide images.

## Suggestions / next steps
- Add ranking diagnostics in `lit_module.py` (Spearman, top-k recall, NDCG) to expose learning failure early.
- Log an RRI proxy on the same scale as labels: map predicted expected class to bin midpoints (`rri_binner.json`).
- Restore richer scene channels (`counts_norm`, `occ_input`, `unknown`, `new_surface_prior`) and/or enable `free_input`.
- Increase frustum depth coverage or make it adaptive; verify WORLD->VOXEL mapping and occ_pr logits semantics.

## Docs updates
- Summarized these findings in `docs/contents/todos.qmd` under "Recent Analysis & Debugging".

## Implementation updates (Dec 30, 2025)
- Added Spearman correlation, confusion matrix, and label histogram tracking to `oracle_rri/oracle_rri/lightning/lit_module.py`.
- Corrected `pred_rri_mean` logging to use a bin-midpoint RRI proxy; preserved the ordinal score as `pred_score_mean`.
- Moved RRI bin midpoint computation into `oracle_rri/oracle_rri/vin/rri_binning.py` (`RriOrdinalBinner.class_midpoints`).
- Registered metrics via `ModuleDict` with non-reserved keys to avoid device mismatch and key collisions; confusion matrix now updates every 10 steps.

## Testing note
- `oracle_rri/tests/integration/test_vin_lightning_real_data.py` currently fails due to `torchsummary` calling `LearnableFourierFeatures.forward(r=...)`. The summary path assumes the old SH encoder signature; consider updating `summarize_vin` or skipping torchsummary in that test.
- After the metrics change, the same test now fails because the summary string no longer contains `torchsummary: Vin` (it reports a VIN v2 summary instead). The assertion in the test likely needs updating.

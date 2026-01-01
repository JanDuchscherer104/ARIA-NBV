# VinModelV2 Feature/Encoding Ideas (2026-01-01)

## Context
- EVL voxel grid is local (default ~4x4x4 m, aligned to last pose), so many candidates fall partially or fully out-of-bounds.
- Current VinModelV2 uses pose R6D+translation + scene field channels (occ_pr/occ_input/counts/unknown/etc.) + pose-conditioned global pooling.

## Suggested Feature Sets
1) **Coverage + occupancy core**
   - Use `occ_pr`, `occ_input`, `free_input`, `counts_norm`, `unknown`, `new_surface_prior`.
   - Add candidate-level `valid_frac` and `center_in_bounds` as explicit features or gating.

2) **Boundary/surface cues**
   - Derived channels: `surface_boundary = occ_pr * free_input`, `uncertainty = occ_pr * (1 - occ_pr)`.
   - Use shallow 3D conv or 1x1x1 projection for lightweight enrichment.

3) **Neck features (compressed)**
   - Add `occ_feat` (and optionally `obb_feat`) with 1x1x1 conv compression (e.g., 64 -> 8/16) to keep compute reasonable.
   - Keep head outputs for interpretability; use neck only if accuracy plateaus.

4) **Entity-aware cues**
   - Use decoded OBBs (`obb_pred`, `obb_pred_probs_full`) to form per-candidate features:
     - distance to nearest OBB center,
     - view alignment w.r.t. OBB axes,
     - fraction of candidate frustum intersecting OBBs,
     - top-k semantic probabilities.

5) **2D token features**
   - Use `feat2d_upsampled` or `token2d` by projecting semidense points into current RGB views, then reproject into candidate view to approximate VIN-NBV's `F_empty` and texture cues.
   - Provides non-voxel features when candidates are outside the EVL grid.

6) **Semidense point cloud cues (voxel-independent)**
   - Candidate-frustum projection of semidense points to compute:
     - projected coverage/empty fraction (`F_empty`),
     - depth statistics (mean/variance),
     - visibility counts per pixel or per angular bin.

## Encoding Schemas
- **Candidate-relative positional keys**: build pos_grid in candidate frame (TODO in `model_v2.py`) so attention keys align with candidate queries.
- **Shell encoding**: bring back SH/LFF shell features (direction u, forward f, radius r, alignment) to encode geometry independent of voxel extent.
- **Distance-to-voxel features**: explicit scalar features for candidate center to voxel center, signed distance to voxel bounds, and in-bounds fraction.
- **Multi-scale pooling**: pool field at 2-3 grid sizes and concatenate; helps if only part of grid is informative.

## Mitigating the 4m grid limit
- Add explicit out-of-bounds indicators and gate reliance on voxel features.
- Use non-voxel features (semidense projections, 2D tokens) as fallback when `valid_frac` is low.
- If feasible, consider multi-anchor EVL inference (multiple grid centers) for wider coverage.

## Documentation
- Added `docs/contents/impl/vin_v2_feature_proposals.qmd` with feature bundles, encoding schemas, and mitigation plan for EVL voxel extent.
- Updated `docs/references.bib` with Wikipedia entries for ordinal regression and attention (Accessed 2026-01-01).

## 2026-01-01 updates
- Expanded `docs/contents/impl/vin_v2_feature_proposals.qmd` with theory + implementation sketches for
  - pose-conditioned attention diagnostics (entropy/peak weights),
  - semidense projection features (always-on) with `EfmPointsView.collapse_points` guidance.
- Added Wikipedia references for view frustum, point cloud, and pinhole camera model in `docs/references.bib`.

## CORAL ordinal-to-regression bridge (design notes)
- Current `coral_logits_to_prob` correctly converts threshold probs P(y>k) to marginals; `lit_module.py` uses `expected_from_probs` for aux regression, which is correct.
- `expected_normalized` in `coral_expected_from_logits` is the expected ordinal label (sum P(y>k)/(K-1)), not a calibrated RRI; avoid using it as a regression target.
- Proposed implementation:
  - Add learnable monotone bin reps `u_k` (initialized from binner bin means/midpoints, parameterized via softplus deltas).
  - Compute `pred_rri = sum_k pi_k * u_k` where `pi_k` are marginals from CORAL.
  - Use Huber loss on `pred_rri`, plus regularizers: `(u_k - mu_k)^2` and optional residual head penalty.
  - Expose attention/metrics to monitor monotonicity and calibration.

## 2026-01-01 CORAL integration updates
- Added `MonotoneBinValues` + bin-value expectation utilities to `oracle_rri/oracle_rri/rri_metrics/coral.py`.
- Extended `CoralLayer` with learnable monotone bin reps and expected-from-probs/logits helpers.
- Added `VinModelV2.init_bin_values` + Lightning init hook (`_maybe_init_bin_values`) to seed bin reps from binner bin means.
- Updated `docs/contents/impl/coral_intergarion.qmd` with implementation details and code snippets.

## 2026-01-01 CORAL diagnostics panel
- Added a new "CORAL / Ordinal" tab to VIN diagnostics in `oracle_rri/oracle_rri/app/panels.py` with:
  - RRI histogram + bin edges overlay,
  - bin representative plots (means/midpoints/learned u_k),
  - per-candidate cumulative and marginal distributions,
  - monotonicity violation histogram,
  - per-candidate CORAL loss histogram.

## 2026-01-01 doc_classifier aux loss
- Added optional aux regression loss (`mae`/`huber`) to `external/doc_classifier/lightning/lit_module.py` with weight control.
- Auxiliary loss operates on softmax probabilities vs one-hot targets and is added to CE when enabled.
- Ruff check reports many pre-existing style violations in that external file; not addressed.

## 2026-01-01 VIN aux loss update
- VIN Lightning now supports `aux_regression_loss = "huber"` (Smooth L1) and defaults to huber.
- Implemented in `oracle_rri/oracle_rri/lightning/lit_module.py` with explicit switch.

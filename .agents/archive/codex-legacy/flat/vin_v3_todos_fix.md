# VINv3 TODO fixes (2026-01-07)

## Changes made
- VIN-Core: removed trajectory encoder support from `VinModelV3Config` and `VinModelV3` (no traj features in head or diagnostics).
- Semidense reliability weighting: enabled `obs_count` sampling, propagated `obs_count` through projection, and implemented `w = a * b` with separate normalization (log1p obs_count + 95th percentile inv_dist_std).
- Semidense injection: removed concatenation into head features; semidense stats now only FiLM-modulate global features.
- Candidate validity: semidense absence no longer invalidates all candidates (vis_frac defaults to 1 when semidense unavailable).
- Pose batching: `ensure_pose_batch` now broadcasts `(1,12)` to `(B,12)` and validates batch sizes.
- Cleanup: removed unused `self.pool` in `PoseConditionedGlobalPool`; trimmed VINv3 diagnostics to drop `traj_feat`.
- Documentation: added FIXMEs about CW90 camera consistency in `model_v3.py` module docstring and clarified `apply_cw90_correction` docstring.

## Tests run
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_types.py oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/integration/test_vin_v3_real_data.py oracle_rri/tests/integration/test_vin_lightning_real_data.py`

## Findings / issues
- `make context` failed due to unresolved merge conflict markers in `oracle_rri/oracle_rri/app/panels/offline_stats.py` (line ~645). Needs cleanup before class diagram generation will work again.

## Open suggestions / follow-ups
- CW90 camera correction remains unimplemented (FIXME): decide whether to rotate `p3d_cameras` or assert corrected cameras.
- If strict “v3-only” modules are desired, consider moving `oracle_rri/oracle_rri/vin/traj_encoder.py` to `vin/experimental` and updating exports/imports accordingly.
- Consider adding a small unit test for weighted semidense visibility (w=a*b) to validate normalization behavior.

## Follow-up (FIXME cleanup)
- Removed non-CW90 FIXMEs in `model_v3.py` by making fallback behavior explicit (uniform reliability when obs_count / inv_dist_std missing; semidense absence neutral) and enforcing pose_vec presence.
- Added fallback sampling in `oracle_rri/oracle_rri/pose_generation/positional_sampling.py` when `power_spherical` is stubbed/unavailable (normalized Gaussian; approximate forward bias when kappa is set).

## Test note
- Integration tests can be affected by test-local stubs for `coral_pytorch` and `power_spherical`. For validation, I pre-imported the real packages before `pytest`:
  - `oracle_rri/.venv/bin/python - <<'PY' ... pytest.main([...])`

## Voxel projection gate
- Added voxel projection features from pooled voxel centers and a FiLM gate on `global_feat`.
- `pts_world` can arrive as flattened `(B,N,3)`; pooling now supports both `(B,N,3)` and `(B,D,H,W,3)` by inferring grid shape and center-cropping before adaptive pooling.
- Diagnostics now include `voxel_proj`.

## Semidense global normalization
- Added global semidense summary stats to `VinModelV3Config` (obs_count + inv_dist_std min/max/p95/mean/std).
- Normalization for semidense reliability now uses these global stats instead of per-camera max/quantile.

## VinSnippetView lengths + cache padding
- Added `lengths` to `VinSnippetView` and plumbed through builders, cache payloads, and batch collation.
- VIN snippet cache now pads `points_world` to 50,000 rows and stores `points_length` for fast slicing.
- Cache metadata/hash updated to include `pad_points`; cache version bumped to 2.
- VINv3 semidense sampling now uses `lengths` to avoid per-batch finite scanning.

## Test note
- `oracle_rri/tests/vin/test_arch_viz.py` fails due to missing `oracle_rri.vin.arch_viz` (pre-existing; not touched). Skipped in final test run.

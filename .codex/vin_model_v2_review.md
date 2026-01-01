# vin_model_v2 review (2025-12-30)

## Context
- Task: review `oracle_rri/oracle_rri/vin/model_v2.py` for learning issues and inspect W&B run `vz1h5q39` history.
- Files reviewed: `oracle_rri/oracle_rri/vin/model_v2.py`, `oracle_rri/oracle_rri/vin/model.py`, `oracle_rri/oracle_rri/vin/backbone_evl.py`, `oracle_rri/oracle_rri/lightning/lit_module.py`.

## Findings (potential causes of weak learning)
1) Pose scaling direction is inconsistent with v1 and the config docstring.
   - v2 divides by learned scale (`pose_vec = [t/scale_t, r6d/scale_r]`), while v1 multiplies.
   - This flips the meaning of `pose_scale_init` and can shrink pose inputs as scales grow.
   - File: `oracle_rri/oracle_rri/vin/model_v2.py` (around `_encode_pose_r6d`).

2) `valid_frac` is effectively binary and does not reflect voxel coverage.
   - v2 sets `valid_frac = candidate_valid`, and `candidate_valid` is only finiteness of pose.
   - Loss weighting (`use_valid_frac_weight`) becomes almost constant and no longer down-weights out-of-grid candidates.
   - Files: `oracle_rri/oracle_rri/vin/model_v2.py` (candidate_valid/valid_frac), `oracle_rri/oracle_rri/lightning/lit_module.py` (loss weighting).

3) `apply_cw90_correction` claims to also correct cameras but `p3d_cameras` is ignored.
   - Docstring says cameras are corrected, but code only rotates poses.
   - If cameras are later used in v2, this will reintroduce the frame mismatch noted in `docs/contents/todos.qmd`.
   - File: `oracle_rri/oracle_rri/vin/model_v2.py` (apply_cw90_correction block).

## Suggestions
- Align pose scaling semantics with v1 (or update docstring + config naming if v2 intentionally uses inverse scaling).
- Compute a real `valid_frac` (e.g., voxel-coverage proxy) or disable `use_valid_frac_weight` for v2 to avoid misleading weighting.
- Either rotate `p3d_cameras` when `apply_cw90_correction=True` or update doc/comments to reflect that v2 ignores cameras.

## W&B notes
- Run `traenslenzor/aria-nbv/vz1h5q39` is marked crashed; `run.history()` returned a sampled pandas DataFrame with many NaNs for missing keys.
- Use `run.scan_history(keys=[...])` for full, non-sampled records when debugging metrics.

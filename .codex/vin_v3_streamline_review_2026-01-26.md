# VINv3 streamline review + sweep comparisons (2026-01-26)

## Sweep comparisons (recap)
- Best Optuna DB trial (#20) in corrected regime: point encoder OFF, traj encoder ON, semidense frustum ON, voxel-valid feature ON, voxel gate OFF.
- Best W&B run `wsfpssd8`: point encoder ON, traj encoder ON, semidense frustum ON, voxel-valid feature ON, voxel gate OFF.
- VINv3 baseline drops traj encoder + semidense frustum + point encoder; semidense only FiLM-modulates global features, reducing per-candidate signal strength relative to best trials.

## Potential issues in VINv3 (prior to fixes)
- CW90 mismatch: poses were corrected but `p3d_cameras` were not, silently corrupting semidense projection features.
- Semidense missing paths: when snippet or camera metadata was absent/mismatched, v3 quietly returned zero/ones features and continued.
- VinSnippetView NaNs: missing/invalid points could propagate, but model masked silently.

## Changes applied to `oracle_rri/oracle_rri/vin/model_v3.py`
- **Fail-fast semantics**:
  - Require snippet inputs (EfmSnippetView or VinSnippetView) for semidense projection.
  - Require semidense points to exist and contain 5 channels (x,y,z,inv_dist_std,obs_count).
  - Reject non-finite XYZ values explicitly.
  - Require valid `p3d_cameras.image_size` and batch alignment; errors instead of returning None.
  - Disallow missing projection data (no zero/ones fallbacks).
- **CW90 explicitness**:
  - If `apply_cw90_correction=True`, now requires `p3d_cameras.cw90_corrected=True` or raises.
- **Streamlining**:
  - Removed the semidense FiLM path (`sem_proj_film`) entirely; semidense stats are now diagnostics/validity only.
  - Refactored voxel projection FiLM into `_apply_film` helper.

## Follow-ups
- Upstream: set `p3d_cameras.cw90_corrected=True` after rotating camera extrinsics, or disable `apply_cw90_correction`.
- If semidense cache stats drift, update v3’s `semidense_obs_count_*` / `semidense_inv_dist_std_*` config values.

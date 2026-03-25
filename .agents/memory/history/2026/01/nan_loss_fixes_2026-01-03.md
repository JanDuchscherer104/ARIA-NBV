---
id: 2026-01-03_nan_loss_fixes_2026-01-03
date: 2026-01-03
title: "Nan Loss Fixes 2026 01 03"
status: legacy-imported
topics: [nan, loss, fixes, 2026, 01]
source_legacy_path: ".codex/nan_loss_fixes_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# NaN loss fixes applied (VIN v2)

## What changed
- Sanitized semidense projection inputs in `oracle_rri/vin/model_v2.py`:
  - Safe `x/y/z` values for invalid points.
  - `torch.nan_to_num` on `x/y/z` and `weights_cam` before aggregation.
  - Depth statistics now use `z_safe` and masked weights.
- Sanitized frustum token inputs (`x_norm/y_norm/depth_m/inv_dist_std`) using safe values.
- Added a guard in `oracle_rri/lightning/lit_module.py` to skip batches with non-finite logits and log `*/skip_nonfinite_logits`.
- Added unit test ensuring NaN safety in semidense projection/frustum features (`tests/vin/test_vin_v2_utils.py`).

## Why
Non-finite depths or weights in invalid projections were propagating through aggregation, contaminating semidense features and producing NaN logits/losses.

## Verification targets
- `train/loss_step`, `train/coral_loss_step`, `train/aux_regression_loss_step` stay finite for sustained steps.
- No `skip_nonfinite_logits` logs after the fix.

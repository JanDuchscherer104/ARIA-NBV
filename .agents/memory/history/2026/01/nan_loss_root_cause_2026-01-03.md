---
id: 2026-01-03_nan_loss_root_cause_2026-01-03
date: 2026-01-03
title: "Nan Loss Root Cause 2026 01 03"
status: legacy-imported
topics: [nan, loss, root, cause, 2026]
source_legacy_path: ".codex/nan_loss_root_cause_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# NaN loss root cause (VIN v2 training)

## Summary
Training still reports `train/loss_step: nan` and all CORAL losses NaN. Masking in `lit_module.py` handles non-finite RRI labels, so the NaNs are likely coming from the model forward path (non-finite logits/probs).

## Likely cause (most direct)
Non-finite values in semidense projection features propagate to logits:
- In `oracle_rri/vin/model_v2.py::_encode_semidense_projection_features`, invalid points are masked via `valid`, but `z` (and `weights_cam`) are still used directly in `depth_mean` / `depth_var`. In PyTorch, `NaN * 0 = NaN`, so any invalid/NaN depth can contaminate the aggregates and propagate to `semidense_proj` → `head_mlp` → logits → CORAL loss.
- The frustum-context path builds tokens from `x/y/z` before masking; invalid `x/y/z` can be NaN and flow into the attention block prior to the `masked_fill`.

## Suggested fix
Sanitize projection tensors before aggregation:
- Replace `x/y/z` with safe values for invalid points (`torch.where(valid, x, 0)` etc.).
- Apply `torch.nan_to_num` (or masked set-to-zero) to `weights_cam` and `z` before computing depth statistics.
- Use the sanitized values for `x_norm/y_norm/depth_m/inv_dist_std` in the frustum tokens.

Optional guard:
- In `lit_module._step`, if `torch.isfinite(logits)` is false, log and skip the batch to avoid poisoning the run; this is diagnostic rather than a final fix.

## Acceptance criteria
- `train/loss_step`, `train/coral_loss_step`, `train/aux_regression_loss_step` remain finite for at least several hundred steps.
- No `skip_nonfinite_logits` (if guard added) after the sanitization patch.

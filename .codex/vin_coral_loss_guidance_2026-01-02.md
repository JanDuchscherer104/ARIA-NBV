# VIN CORAL loss guidance (2026-01-02)

- Recommended baseline for current collapse: `coral_loss_variant="balanced_bce"` with `coral_bias_init="prior_logits"` and binner-derived priors (`bin_counts`).
- Keep aux regression as a secondary term with decay (e.g., weight=10, gamma=0.90, min=0.1); do not sum CORAL and balanced/focal losses since they target the same thresholds.
- If collapse persists, switch to `coral_loss_variant="focal"` (gamma ~2) as a stronger anti-collapse option.
- NaN losses likely come from non-finite `rri` values; current `_step` uses raw `rri` in aux loss and logging. Reintroduce a finite-mask gate (skip batch if none valid) to prevent NaN propagation.
- Ensure fitted binner JSON includes `bin_counts`; refit if missing so priors are meaningful.

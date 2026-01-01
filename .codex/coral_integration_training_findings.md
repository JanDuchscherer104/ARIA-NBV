# CORAL integration: training dynamics notes

## What was added

Added a new “Training dynamics, monitoring, and two practical fixes” section to
`docs/contents/impl/coral_intergarion.qmd` that captures:

- Why `aux_regression_loss` can drop extremely fast relative to `coral_loss`
  (loss-scale mismatch + mean/median-optimal behavior early).
- Fix #1: introduce an explicit `λ_aux` (aux regression weight) instead of the
  current implicit `λ_aux = 1` from `combined_loss = coral_loss + aux_loss`.
- Fix #2: configure `ReduceLROnPlateau` to monitor `val/coral_loss` instead of
  `train/loss` so LR changes respond to the hard objective.
- Sanity checks: chance-level baseline `(K-1) log 2` and interpretation of
  central-band confusion matrices.
- Reminder that learnable `u_k` should be *regularized* (tethered to empirical
  means) in the training objective, not just initialized.

## Verification

- `cd docs && quarto render contents/impl/coral_intergarion.qmd --to html` succeeds.


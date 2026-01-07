# Optuna sweep fix (2026-01-05)

## Issue
Optuna trials failed with `init_values must have shape (K,), got (15,) for K=8.`
Root cause: optuna objective path did not call `_ensure_binner_matches_num_classes`, so an existing 15-class binner was used while `num_classes=8`.

## Fix
Added a binner check/refit inside the optuna objective before `trainer.fit`:
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py` now calls
  `_ensure_binner_matches_num_classes(datamodule=...)` in `objective()`.

## Outcome
Optuna trials should now refit the binner to the trial’s `num_classes` and avoid CORAL init mismatch.

## Notes
- This can overwrite `binner_path` when a mismatch is detected (expected behavior).
- If you want to avoid refitting, ensure `binner_path` already matches `num_classes`.

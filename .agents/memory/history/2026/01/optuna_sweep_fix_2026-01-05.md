---
id: 2026-01-05_optuna_sweep_fix_2026-01-05
date: 2026-01-05
title: "Optuna Sweep Fix 2026 01 05"
status: legacy-imported
topics: [optuna, sweep, 2026, 01, 05]
source_legacy_path: ".codex/optuna_sweep_fix_2026-01-05.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

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

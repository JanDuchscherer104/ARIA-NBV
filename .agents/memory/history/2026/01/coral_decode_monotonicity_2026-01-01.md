---
id: 2026-01-01_coral_decode_monotonicity_2026-01-01
date: 2026-01-01
title: "Coral Decode Monotonicity 2026 01 01"
status: legacy-imported
topics: [coral, decode, monotonicity, 2026, 01]
source_legacy_path: ".codex/coral_decode_monotonicity_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# CORAL decoding + monotonicity logging (2026-01-01)

## Summary
- Added CORAL decoding helpers in `oracle_rri/oracle_rri/rri_metrics/coral.py`:
  - `coral_logits_to_label` (threshold-count decode).
  - `coral_monotonicity_violation_rate` (fraction of P(y>k) increases).
- Updated `oracle_rri/oracle_rri/lightning/lit_module.py`:
  - Confusion-matrix labels now use CORAL threshold decoding instead of argmax.
  - Logs `train/val/test/coral_monotonicity_violation_rate` each step/epoch.

## Rationale
- Argmax on class probs from CORAL logits is unstable when logits are non-monotone, often collapsing to the last class.
- Monotonicity violations are a direct diagnostic of invalid CORAL probability ordering.

## Open suggestions
- Consider monotone projection of `P(y>k)` when forming class probabilities (e.g., reverse cumulative min) if you want stable `coral_logits_to_prob`.
- Optionally log an additional metric: fraction of samples with any violation (binary indicator).
- Consider class-imbalance weighting (per-epoch label histogram) in `coral_loss` if label distribution is skewed.

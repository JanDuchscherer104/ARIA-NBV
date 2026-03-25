---
id: 2026-01-29_vin_diag_bin_values_tab_2026-01-29
date: 2026-01-29
title: "Vin Diag Bin Values Tab 2026 01 29"
status: legacy-imported
topics: [diag, bin, values, tab, 2026]
source_legacy_path: ".codex/vin_diag_bin_values_tab_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

## VIN diagnostics: bin-value comparison tab (2026-01-29)

### Goal
Add a dedicated VIN diagnostics tab that compares:
- fitted binner thresholds (`edges`) and per-class centers (`bin_means` / `midpoints`)
- learned monotone bin representatives `u_k` used by `CoralLayer.expected_from_probs`
- the per-class differences (`learned_u - baseline`)

This is meant to debug “expected RRI” drift after resuming from checkpoints.

### Implementation
- New Streamlit tab: `Bin Values`
  - File: `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/bin_values.py`
  - Reads:
    - `state.module._binner` (`RriOrdinalBinner`)
    - `state.module.vin.head_coral.bin_values.values()` when available
  - Shows:
    - table + plot of `edges` (K-1)
    - table + plot of centers (`midpoints`, optional `bin_mean`) + `learned_u` (K)
    - bar plot of `learned_u - baseline` where baseline is the init target
      (`bin_means` if present else `class_midpoints`)
    - summary stats: mean/max |Δ| and min spacing for learned/baseline
- Wiring:
  - Added `render_bin_values_tab` to `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/__init__.py`
  - Added the new tab to `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`

### Tests
- Added a small unit test for the payload builder helper:
  - `tests/app/panels/test_vin_bin_values_tab.py`
  - Verifies baseline selection (bin means vs midpoints) and presence of diff column.

### Notes / follow-ups
- If we want to debug drift “in expectation space” (not only `u_k`), a natural next plot is:
  `E[RRI]_learned - E[RRI]_binner` across candidates for the current batch (`pred.prob`),
  plus a scatter against `oracle_rri`.

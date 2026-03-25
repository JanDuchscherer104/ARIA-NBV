---
id: 2026-01-02_binner_auto_refit_2026-01-02
date: 2026-01-02
title: "Binner Auto Refit 2026 01 02"
status: legacy-imported
topics: [binner, auto, refit, 2026, 01]
source_legacy_path: ".codex/binner_auto_refit_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Binner auto-refit on num_classes mismatch (2026-01-02)

## Summary
- Added an auto-refit check before training that rebuilds the RRI ordinal binner when `num_classes` differs from the saved binner JSON.
- `fit_binner_and_save(...)` now accepts an optional datamodule and an `overwrite` flag to reuse existing loaders and overwrite the binner when required.
- Documented the auto-refit behavior in `docs/contents/impl/vin_nbv.qmd`.
- Added a unit test to validate the auto-refit flow with a dummy datamodule.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/lightning/test_binner_auto_refit.py` (pass)
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py -k fit_runs_real_data_smoke` (fail)
  - Failure: `OracleRRIConfig` rejects `candidate_chunk_size` as an extra field (ValidationError). This appears pre-existing and unrelated to the binner changes.

## Open Issues / Suggestions
- Investigate why `OracleRRIConfig` no longer accepts `candidate_chunk_size` in the integration test. Either remove that argument in the test or reintroduce the field in the config (if still intended).
- Consider logging or alerting when `rri_binner_fit_data.pt` is missing and auto-refit must re-sample from the dataset.

---
id: 2026-01-02_2026-01-02+testing_attribution_panel
date: 2026-01-02
title: "Testing Attribution Panel"
status: legacy-imported
topics: [testing, attribution, panel]
source_legacy_path: ".codex/2026-01-02+testing_attribution_panel.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Testing & Attribution panel update (2026-01-02)

## Changes
- Moved VIN head attribution explorer out of `wandb.py` into new panel `oracle_rri/oracle_rri/app/panels/testing_attribution.py`.
- Added new Streamlit page **Testing & Attribution** in `oracle_rri/oracle_rri/app/app.py` and export in `oracle_rri/oracle_rri/app/panels/__init__.py`.
- Removed UI controls for checkpoint path override and cache map_location; cache loads always use CPU.
- Implemented robust checkpoint loading using `torch.load` + `VinLightningModuleConfig` reconstruction, avoiding `VinLightningModule.load_from_checkpoint` config error.
- Updated docs page list in `docs/contents/impl/data_pipeline_overview.qmd` to include W&B Analysis and Testing & Attribution.

## Tests run
- `python -m pytest tests/vin/test_vin_model_v2_integration.py -m integration`

## Notes
- Attribution device selection still available; module weights are loaded on CPU then moved to the chosen device.
- If we want end-to-end attributions through the full VIN pipeline, we’ll need to keep gradients from earlier modules instead of using detached `debug.feats`.

## Follow-up
- Added attribution source selector (VIN Diagnostics batch vs offline cache).
- When using VIN Diagnostics, the panel no longer demands a cache and reuses the last batch.
- Cache map_location remains fixed to CPU; offline cache UI only shows when that source is selected.
- Re-ran integration test: `tests/vin/test_vin_model_v2_integration.py -m integration`.

## VIN diagnostics fix
- Fixed typo in `vin_diag_tabs/summary.py` (`ctx.cdfg` -> `ctx.cfg`) that caused the VIN Diagnostics page to crash.
- Re-ran integration test: `tests/vin/test_vin_model_v2_integration.py -m integration`.

## VIN diagnostics checkpoint support
- Added checkpoint selection to VIN Diagnostics sidebar and module override via `_load_vin_module_from_checkpoint`.
- Shared checkpoint loader implemented in `vin_utils.py` and reused in the Testing & Attribution panel.
- Re-ran integration test: `tests/vin/test_vin_model_v2_integration.py -m integration`.

## Attribution expected score
- Attribution now targets and displays `head_coral.expected_from_probs(pred.prob)` when bin values are initialized.
- Falls back to normalized expected score with a warning if bin values are missing.
- Label updated to “Pred RRI (expected)”.
- Re-ran integration test: `tests/vin/test_vin_model_v2_integration.py -m integration`.

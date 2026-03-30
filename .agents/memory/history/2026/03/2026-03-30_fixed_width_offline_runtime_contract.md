---
id: 2026-03-30_fixed_width_offline_runtime_contract
date: 2026-03-30
title: "Fixed-Width Offline Runtime Contract"
status: done
topics: [data-handling, offline-store, lightning, testing]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_types.py
  - aria_nbv/aria_nbv/lightning/lit_module.py
  - aria_nbv/aria_nbv/data_handling/efm_views.py
  - aria_nbv/aria_nbv/app/panels/vin_utils.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/lightning/test_vin_batch_collate.py
  - aria_nbv/tests/vin/test_vin_model_v3_methods.py
---

Task: implement the fixed-width offline runtime contract requested during PR review follow-up, keep block-backed offline samples at store width with `candidate_count`, and resolve the side regressions in snippet-view detection and VIN panel config construction.

Method: threaded `candidate_count` through `VinOfflineOracleBlock` and `VinOracleBatch`, kept offline oracle tensors full-width on read, repaired padded pose/camera rows in memory so the full-width runtime path stays model-safe, updated batching and candidate shuffling to preserve padded tails, switched Lightning masking from “finite RRI only” to `candidate_count` plus finite checks, replaced descriptor-invoking `hasattr` probes with `inspect.getattr_static`, and moved `vin_utils.py` to `aria_nbv.data_handling` imports.

Findings/outputs: the offline reader can now support fixed-width batching without losing the stored candidate budget, padded candidates are excluded from loss/metrics by `candidate_count`, shuffling no longer pulls padded tail entries into the valid prefix, the legacy EFM snippet compatibility check no longer triggers property evaluation, and the VIN diagnostics helper now builds source configs from the same module family that `VinDataModuleConfig` validates.

Verification:
- `aria_nbv/.venv/bin/ruff check aria_nbv/aria_nbv/data_handling/vin_oracle_types.py aria_nbv/aria_nbv/data_handling/_offline_dataset.py aria_nbv/aria_nbv/lightning/lit_module.py aria_nbv/aria_nbv/data_handling/efm_views.py aria_nbv/aria_nbv/app/panels/vin_utils.py aria_nbv/tests/data_handling/test_vin_offline_store.py aria_nbv/tests/lightning/test_vin_batch_collate.py aria_nbv/tests/vin/test_vin_model_v3_methods.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/data_handling/test_vin_offline_store.py tests/lightning/test_vin_batch_collate.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/vin/test_vin_model_v3_methods.py tests/vin/test_vin_utils.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/data_handling/test_public_api_contract.py`

Canonical state impact: none. This work changes runtime implementation details and regression coverage, but not the maintained project-state or decision documents.

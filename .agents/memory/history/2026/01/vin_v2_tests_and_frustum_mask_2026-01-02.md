---
id: 2026-01-02_vin_v2_tests_and_frustum_mask_2026-01-02
date: 2026-01-02
title: "Vin V2 Tests And Frustum Mask 2026 01 02"
status: legacy-imported
topics: [v2, tests, frustum, mask, 2026]
source_legacy_path: ".codex/vin_v2_tests_and_frustum_mask_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

## Summary

- Added unit coverage for semidense projection stats and frustum MHCA behavior in VIN v2.
- Fixed a NaN hazard in semidense frustum attention when all projected points are invalid.

## Findings / Fixes

- **NaN hazard**: `MultiheadAttention` can produce NaNs when every key token is masked (all invalid points).  
  - **Fix**: For rows with no valid points, clear the padding mask so attention runs over zeroed tokens, then zero the output via `valid_any` masking.

## Tests Run

- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_utils.py tests/vin/test_vin_model_v2_integration.py`

## Suggestions / Follow-ups

- Consider a small helper to normalize the projection + frustum pipeline in tests (shared fixtures), if this grows further.

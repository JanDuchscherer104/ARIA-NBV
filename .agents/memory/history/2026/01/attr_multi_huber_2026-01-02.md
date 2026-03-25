---
id: 2026-01-02_attr_multi_huber_2026-01-02
date: 2026-01-02
title: "Attr Multi Huber 2026 01 02"
status: legacy-imported
topics: [attr, multi, huber, 2026, 01]
source_legacy_path: ".codex/attr_multi_huber_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Attribution: huber-based candidate selection + multi-sample aggregation

## Summary
- Added candidate selection modes based on per-candidate Huber error (min/max) between predicted and oracle RRI.
- Added multi-sample attribution aggregation for offline cache (contiguous sample range).
- Added optional attribution std display and candidate index list for aggregated runs.

## Files changed
- `oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Notes
- Multi-sample aggregation is enabled only for offline cache sources and uses contiguous indices starting at `Cache index`.
- Huber uses PyTorch SmoothL1 with default beta=1.0, matching training.

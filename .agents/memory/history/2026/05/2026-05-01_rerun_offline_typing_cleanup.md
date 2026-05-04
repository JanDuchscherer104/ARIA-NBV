---
id: 2026-05-01_rerun_offline_typing_cleanup
date: 2026-05-01
title: "Rerun Offline Typing Cleanup"
status: done
topics: [rerun, vin-offline, typing, simplification]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_metadata.py
  - aria_nbv/aria_nbv/rerun_inspector/_sample.py
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/data_handling/_offline_visual_inventory.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_types.py
---

## Task

Cleaned up the typing and over-defensive access introduced with the compact offline modality and Rerun inspector work.

## Method

Imported concrete `VinOfflineSample` and dataset types from their owning modules instead of through the lazy package root. Replaced compact-modality `getattr` access with direct typed field access. Tightened Rerun module typing through a protocol, added `numpy.typing` annotations, and converted OBB/probability writer helpers to typed `ObbTW | Tensor` and `Tensor | Sequence[Tensor]` contracts.

## Verification

Ran Ruff format/check on touched files. Ran strict mypy on the Rerun inspector source files, which passed. Ran the focused Rerun/offline/collate pytest set, which passed with 50 tests and the existing warnings.

## Canonical State Impact

No canonical memory updates needed.

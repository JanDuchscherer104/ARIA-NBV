---
id: 2026-05-01_rerun_offline_modalities
date: 2026-05-01
title: "Rerun Offline Modalities"
status: done
topics: [vin-offline, rerun, data-handling]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_types.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/lightning/test_vin_batch_collate.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
assumptions:
  - GT meshes remain out of the VIN offline store and are loaded only through live EFM/processed-mesh attachment for Rerun inspection.
---

## Task

Implemented the Rerun + offline modality upgrade while honoring the corrected scope that meshes are Rerun-only diagnostics and not persisted training data.

## Method

Added compact optional VIN offline blocks for GT OBBs, detected OBBs/probabilities, and trajectory timing/gravity. Extended `VinOfflineSample`, `VinOracleBatch`, source configuration, visual inventory, and Rerun logging around those compact numeric fields. Kept raw RGB/depth keyframes and meshes tied to live EFM attachment.

## Outputs

The immutable format version was bumped. The writer now emits `gt.obbs`, `detected.obbs`, `detected.obb_probs`, semantic-map records, and `vin.trajectory.*` when available. The reader and batch collation reconstruct typed compact blocks and reject inconsistent semantic maps. Rerun logs GT mesh from attached EFM/processed mesh only, plus GT/detected OBB line strips, trajectory line strips, cached candidate depth images, and optional live RGB/depth keyframes.

## Verification

Ran `uv run ruff format` and `uv run ruff check` on touched Python files. Ran `uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py tests/data_handling/test_vin_offline_store.py tests/lightning/test_vin_batch_collate.py -q`, which passed with 50 tests and two pre-existing warnings.

## Canonical State Impact

No canonical memory updates needed. Existing offline stores need rebuilding to contain the new compact OBB/trajectory blocks; Rerun can still inspect meshes through live EFM attachment.

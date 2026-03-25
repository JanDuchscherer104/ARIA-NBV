---
id: 2025-12-24_model_checkpoint_step_based_2025-12-24
date: 2025-12-24
title: "Model Checkpoint Step Based 2025 12 24"
status: legacy-imported
topics: [model, checkpoint, step, based, 2025]
source_legacy_path: ".codex/model_checkpoint_step_based_2025-12-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# ModelCheckpoint: step-based checkpointing (2025-12-24)

## Motivation

- For very large datasets, a single epoch can take “forever”, so epoch-bound checkpointing (the Lightning default) is not sufficient.

## What changed

- `oracle_rri/oracle_rri/lightning/lit_trainer_callbacks.py`
  - Added optional `TrainerCallbacksConfig` fields to control *when* `ModelCheckpoint` fires:
    - `checkpoint_every_n_train_steps`
    - `checkpoint_train_time_interval` (dict -> `timedelta(**...)`)
    - `checkpoint_every_n_epochs`
    - `checkpoint_save_last`
    - `checkpoint_save_on_train_epoch_end`
  - Added validation that the schedule knobs are mutually exclusive (Lightning 1.9.5 enforces this too).
  - Updated the default filename template to include formatted loss: `train-loss={train/loss:.4f}`.

## How to use

- For long epochs, prefer step-based snapshots:
  - `checkpoint_every_n_train_steps: 1000`
  - optionally `checkpoint_save_last: true`
- Alternatively use wall-clock:
  - `checkpoint_train_time_interval: { minutes: 10 }`

## Test coverage

- Added `oracle_rri/tests/lightning/test_trainer_callbacks_model_checkpoint.py` to ensure the config wires these fields into `ModelCheckpoint` correctly and rejects invalid combinations.

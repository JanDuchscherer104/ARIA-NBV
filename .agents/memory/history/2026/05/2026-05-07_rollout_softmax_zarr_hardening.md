---
id: 2026-05-07_rollout_softmax_zarr_hardening
date: 2026-05-07
title: "Rollout Softmax Zarr Hardening"
status: done
topics: [rollouts, zarr, q-h, data, litkg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/aria_nbv/pose_generation/rollout_trace.py
  - aria_nbv/tests/data_handling/test_rollout_zarr_store.py
  - .agents/issues.toml
  - .agents/resolved.toml
  - .agents/memory/state/DECISIONS.md
  - .agents/work/research-review-proposal-distillation/rollout-qh-zarr-dto-digest.md
---

## Task

Implemented and hardened the multi-step oracle softmax rollout tracer bullet
around standalone `rollouts.zarr` storage, then resolved the review blocker TODO
for the four training-facing defects found in the first Zarr implementation.

## Outputs

- `rollouts.zarr` writer validation now protects explicit target-RRI provenance,
  multi-target target identity through `q_h`, PoseTW root-relative transforms,
  per-rollout split/lineage fields, and aligned candidate row-table lengths.
- `RolloutTrace` lineage and step DTOs carry the selected-action and
  temperature-softmax provenance needed by the first `Q_H` replay store.
- Regression tests cover non-oracle score fallback, invalid label masking,
  multi-target identity, root-relative pose transforms, lineage/split fields,
  and non-synthetic lineage validation.
- Durable rollout decisions were added to `DECISIONS.md`.
- The ignored rollout/Q_H/Zarr DTO digest was ingested through litkg; the local
  route query returns `rollout-qh-zarr-dto-digest` as relevant evidence.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_rollout_zarr_store.py aria_nbv/pose_generation/rollout_trace.py tests/data_handling/test_rollout_zarr_store.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- ingest --config .configs/litkg.toml`
- `make kg-route KG_TASK="rollout Q_H Zarr DTO temperature-softmax requirements" KG_FORMAT=json`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

`make kg-ollama-check` failed because no process was reachable on
`127.0.0.1:11434` from this workspace, even though the litkg CLI ingest path
completed successfully.

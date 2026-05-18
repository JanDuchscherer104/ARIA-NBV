---
id: 2026-05-18_full_scale_generation_readiness
date: 2026-05-18
title: "Full-Scale Generation Readiness Implementation"
status: done
topics: [rollouts, q_h, selected_depth, lrz, data_generation]
confidence: high
canonical_updates_needed: []
artifacts:
  - /tmp/aria-nbv-rollout-smoke-qh-source-20260518/shard-000000
  - /tmp/aria-nbv-rollout-smoke-qh-20260518-shards.jsonl
  - /tmp/aria-nbv-rollout-smoke-qh-source-20260518-status.json
---

## Task

Implement the full-scale offline dataset generation readiness plan without launching a blind full-scale run. The local goal was to make the rollout store schema/docs coherent enough for a cluster smoke and to prove one isolated local shard write path.

## Output

- `rollouts.zarr` schema `0.5-selected-depth` now persists a derived `q_h/` training-hot view alongside canonical factual tables.
- `q_h/` stores state ids, source/target ids, candidate ids, masks, one-step target/scene RRI, invalid reasons, TD selected-action fields, terminal fields, and discount arrays with configurable state-row chunking.
- Validation compares persisted `q_h/` arrays back to derived factual-table values and rejects missing or divergent arrays.
- The rollout README, Typst data-layout trees, Mermaid layout diagrams, and agents DB were aligned to `selected_depth/` plus persisted `q_h/`.
- A CUDA/CPU device mismatch in actor-visible target selection was fixed by moving the reference pose to the OBB tensor device before PoseTW composition.
- `SelectedDepthRetentionConfig.renderer_config()` now sets exact selected-depth render size atomically, avoiding Pydantic half-size assignment validation failures.

## Smoke Evidence

The stale local `.data/offline_cache/rollouts_v1_smoke.zarr` still fails validation as expected: schema `0.4-manifested-shards`, missing `selected_depth/`, and missing `q_h/`.

The isolated local shard smoke wrote `/tmp/aria-nbv-rollout-smoke-qh-source-20260518/shard-000000` with `_owner.json`, `_SUCCESS.json`, schema `0.5-selected-depth`, `selected_depths=10`, `q_h_states=10`, and validation `ok=true`.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/rollouts/zarr_store.py aria_nbv/rollouts/dataset_writer.py aria_nbv/data_handling/_target_selection.py tests/rollouts/test_zarr_store.py tests/rollouts/test_dataset_writer.py tests/data_handling/test_target_selection.py`
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_target_selection.py tests/data_handling/test_vin_offline_store.py tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run nbv-plan-rollout-shards --config-path ../.configs/build_rollouts_v1_smoke.toml --rows-per-shard 1 --output-manifest /tmp/aria-nbv-rollout-smoke-qh-20260518-shards.jsonl`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --shard-manifest /tmp/aria-nbv-rollout-smoke-qh-20260518-shards.jsonl --shard-id shard-000000 --output-tmp /tmp/aria-nbv-rollout-smoke-qh-source-20260518/shard-000000.tmp --output-final /tmp/aria-nbv-rollout-smoke-qh-source-20260518/shard-000000`
- `cd aria_nbv && uv run nbv-rollouts-info --store /tmp/aria-nbv-rollout-smoke-qh-source-20260518/shard-000000 --validate --json`
- `cd aria_nbv && uv run nbv-status-rollout-shards --shard-manifest /tmp/aria-nbv-rollout-smoke-qh-20260518-shards.jsonl --final-root /tmp/aria-nbv-rollout-smoke-qh-source-20260518 --output-json /tmp/aria-nbv-rollout-smoke-qh-source-20260518-status.json`
- `bash -n scripts/templates/lrz/rollout_generation.sbatch scripts/templates/lrz/rollout_generation_dry_run.sbatch`
- `aria_nbv/.venv/bin/python tools/mermaid/scripts/aria_mermaid_lint.py docs/figures/diagrams/data_handling/mermaid/offline_rollout_physical_layout.mmd docs/figures/diagrams/data_handling/mermaid/rollout_generation_sequence.mmd docs/figures/diagrams/data_handling/mermaid/data_store_architecture.mmd docs/figures/diagrams/data_handling/mermaid/multi_step_sample_tree.mmd`
- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor_distillation.pdf --root .`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`

## Remaining Blockers

This does not make full-scale generation safe by itself. The next external gate remains one real LRZ/Pyxis shard on DSS-backed paths from the cluster clone, followed by source-store coverage/M1 evidence and a Q_H dataloader throughput benchmark against the persisted `q_h/` view.

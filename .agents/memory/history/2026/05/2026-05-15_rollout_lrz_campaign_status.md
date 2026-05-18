---
id: 2026-05-15_rollout_lrz_campaign_status
date: 2026-05-15
title: "Rollout LRZ Campaign Status"
status: done
topics: [rollouts, lrz, sharding, agents-db]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/AGENTS_INTERNAL_DB.md
  - .agents/todos.toml
  - .configs/build_rollouts_v1_lrz.template.toml
  - .configs/lrz/README.md
  - aria_nbv/aria_nbv/rollouts/cli.py
  - aria_nbv/aria_nbv/rollouts/shards.py
  - scripts/templates/lrz/rollout_generation.sbatch
---

## Task

Implement the accepted issue-resolution order for multi-step offline rollout
samples and make the first LRZ/resumable-generation blocker more concrete.

## Outcome

- Persisted the rollout/Q_H critical path in `.agents/AGENTS_INTERNAL_DB.md` so
  future agents do not follow generic high-priority sorting for this work.
- Added rollout shard campaign status reporting via
  `nbv-status-rollout-shards`, backed by manifest-driven succeeded, failed,
  incomplete, and missing shard states.
- Added a real LRZ rollout array template,
  `scripts/templates/lrz/rollout_generation.sbatch`, that invokes
  `nbv-build-rollouts` for one deterministic shard per Slurm array task.
- Added `.configs/build_rollouts_v1_lrz.template.toml` to force explicit
  DSS-backed rollout/VIN paths instead of relying on local `.data` defaults.

## Verification

- `uv run --project aria_nbv ruff check aria_nbv/aria_nbv/rollouts/cli.py aria_nbv/aria_nbv/rollouts/shards.py aria_nbv/aria_nbv/rollouts/__init__.py aria_nbv/tests/rollouts/test_dataset_writer.py`
- `cd aria_nbv && uv run pytest tests/rollouts/test_dataset_writer.py`
- `cd aria_nbv && uv run pytest tests/rollouts`
- `cd aria_nbv && uv run nbv-plan-rollout-shards --config-path ../.configs/build_rollouts_v1_smoke.toml --rows-per-shard 1 --output-manifest /tmp/rollout_shards.jsonl`
- `cd aria_nbv && uv run nbv-status-rollout-shards --shard-manifest /tmp/rollout_shards.jsonl --final-root /tmp/rollout_shards --output-json /tmp/rollout_shards_status.json`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `bash -n scripts/templates/lrz/rollout_generation.sbatch scripts/templates/lrz/rollout_generation_dry_run.sbatch`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`

## Remaining Blocker

The real LRZ shard template has not been submitted on LRZ. The next production
gate is a one-row live `VinOfflineSample` smoke that writes a real rollout shard
under DSS, followed by `nbv-status-rollout-shards --require-complete`.

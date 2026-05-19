---
id: 2026-05-19_rollout_global_target_rows_rerun_repair
date: 2026-05-19
title: "Rollout Global Target Rows And Rerun Target Repair"
status: done
topics: [rollouts, rerun, target-rri, data-handling]
confidence: high
canonical_updates_needed: []
artifacts:
  - .data/offline_cache/rollouts_v1_realistic.zarr
  - .artifacts/rerun/rollouts_v1_realistic_schema08_idx003.rrd
---

## Task

Fix rollout target lineage and Rerun diagnostics after finding that selector-local
target ids were treated as store-global row ids in realistic rollout stores.

## Method

The rollout Zarr schema was bumped to `0.8-global-target-rows`. The writer now
normalizes target rows into dense store-local ids keyed by source and target
identity, while preserving selector-local ids in `targets/target_source_index`.
Validation now rejects target rows referenced by multiple source rows and flags
structured target ids that name a different snippet than the rollout row.

Rerun rollout logging now emits a visible per-chain target group under
`world/rollout/rollout_XXXXXX/chain_XXXXXX/target/`, including an actor-visible
OBB, center, and JSON metadata. Candidate cameras now include sampling strategy,
mixture, sampler probability, and stable target-RRI rank metadata.

## Outputs

Regenerated `/home/jd/repos/ARIA-NBV/.data/offline_cache/rollouts_v1_realistic.zarr`
with schema `0.8-global-target-rows`: 24 rollout rows, 40 steps, 2400 candidate
rows, 4 globally unique target rows, and 40 selected-depth rows at `240x240`.

Saved `/home/jd/repos/ARIA-NBV/.artifacts/rerun/rollouts_v1_realistic_schema08_idx003.rrd`.
The RRD printout includes the target overlay at
`/world/rollout/rollout_000003/chain_000001/target/`, including
`actor_visible_obb`, `matched_gt_obb`, `center`, and metadata. Selected camera
metadata fields include `sampling_strategy_id`, `sampling_strategy_name`,
`mixture_id`, `sampler_probability`, `target_rri_rank`,
`target_rri_rank_total`, and `target_rri_rank_semantics`.

## Verification

- `cd aria_nbv && uv run pytest tests/rollouts/test_zarr_store.py tests/rollouts/test_dataset_writer.py -q`
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_rollout_zarr_logger.py tests/rerun_inspector/test_loggers.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rollouts aria_nbv/rerun_inspector tests/rollouts tests/rerun_inspector tests/rollout_fixtures.py`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml --dry-run`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml`
- `cd aria_nbv && uv run nbv-rollouts-info --store /home/jd/repos/ARIA-NBV/.data/offline_cache/rollouts_v1_realistic.zarr --validate --json`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_realistic.zarr --rollout-index 3 --rollout-context required --save ../.artifacts/rerun/rollouts_v1_realistic_schema08_idx003.rrd`

## Notes

Matched GT OBB geometry is still not persisted in `rollouts.zarr`; the dedicated
matched-GT overlay is derived from VIN context when rollout context logging is
enabled. `/world/gt/obbs` remains the full GT context layer.

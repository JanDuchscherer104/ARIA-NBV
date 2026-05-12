---
id: 2026-05-12_rollout_core_ruthless_simplification
date: 2026-05-12
title: "Rollout Core Ruthless Simplification"
status: done
topics: [aria-nbv, rollouts, data-handling, zarr, simplification]
confidence: high
canonical_updates_needed: []
---

## Task

Implemented the 2026-05-12 rollout-core simplification request: collapse the
public rollout trace hierarchy into `RolloutZarrRecord`, keep factual
`rollouts.zarr` tables only, remove synthetic-store inspector compatibility
paths, and make target selection consume `VinOfflineSample` roots directly.

## Method

Pruned `RolloutTrace`, `RolloutStepTrace`, trace serialization, and
`RolloutZarrStoreWriter`. The Zarr writer now consumes
`CounterfactualRolloutResult` plus `RolloutLineage`; the derived `q_h_view()`
keeps selected-transition fields and no dense all-action oracle-Q target
placeholders. Rerun rollout logging now displays stored poses directly and uses
normal rollout lineage/context lookup without synthetic root alignment.

## Verification

- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_target_selection.py tests/pose_generation/test_counterfactuals.py tests/rerun_inspector tests/app/panels -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run ruff check ...`
- `cd aria_nbv && uv run ruff check --select F401,TCH ...`

## Canonical State Impact

No state-file update is required. The current rollout contract is now
standalone Zarr replay records over immutable VIN offline rows; old trace DTOs
and MessagePack smoke storage are no longer active production surfaces.

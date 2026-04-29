---
id: 2026-04-29_oracle_pipeline_simplification_slice
date: 2026-04-29
title: "Oracle Pipeline Simplification Slice"
status: done
topics: [oracle, rendering, rri, offline-store, simplification]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - README.md
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/refactors.toml
  - .agents/resolved.toml
  - .agents/memory/state/PROJECT_STATE.md
  - aria_nbv/aria_nbv/rendering/candidate_depth_renderer.py
  - aria_nbv/aria_nbv/rendering/candidate_pointclouds.py
  - aria_nbv/aria_nbv/rendering/unproject.py
  - aria_nbv/aria_nbv/rri_metrics/oracle_rri.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
---

## Task

Implement the most urgent agents DB items after triage: dirty-worktree handoff,
`offline_only.toml` smoke, and the first oracle pipeline simplifications.

## Method

Classified the dirty worktree, ran the `offline_only.toml` summary smoke, ran
M1 contract tests, and implemented the smallest behavior-preserving oracle cuts:
single render budget owner, one public PyTorch3D unprojection path, strict
tensor-native mesh cropping, and default-off rich offline diagnostic payloads.

## Findings

`nbv-summary --config-path offline_only.toml` resolves `.configs/offline_only.toml`
and reaches the offline reader, but fails because
`.data/offline_cache/vin_offline/manifest.json` is absent. This is tracked as
`issue-014`; `todo-008` remains open.

## Outputs

- Removed depth-renderer oversampling and the unused post-render private filter.
- Moved vectorized candidate depth backprojection to
  `rendering.unproject.backproject_depths_p3d_batch`.
- Made empty oracle AABB crops fail explicitly instead of falling back to the
  full mesh.
- Made full DTO diagnostic records opt-in through
  `VinOfflineWriterConfig.include_diagnostic_payloads`.
- Resolved `refactor-005`, `refactor-006`, `refactor-007`, and `refactor-009`.

## Verification

- `cd aria_nbv && uv run ruff format ...`
- `cd aria_nbv && uv run ruff check ...`
- `cd aria_nbv && uv run pytest tests/rendering/test_depth_backprojection_conventions.py tests/rendering/test_candidate_renderer_integration.py tests/rendering/test_pytorch3d_renderer.py`
- `cd aria_nbv && uv run pytest tests/rri_metrics/test_oracle_rri_chunking.py tests/rri_metrics`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py tests/app/panels/test_rl_panel.py tests/vin/test_vin_utils.py`
- `make agents-db AGENTS_ARGS='validate'`
- `make context-contracts`
- `make check-agent-memory`
- `cd docs && quarto render contents/todos.qmd`
- `scripts/nbv_qmd_outline.sh --compact`

## Canonical State Impact

Updated `PROJECT_STATE.md` for the missing offline store blocker and the new
oracle/offline-store contract truth.

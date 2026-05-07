---
id: 2026-05-07_top_k_target_selector
date: 2026-05-07
title: "Top-K Actor-Visible Target Selector"
status: done
topics: [target-selection, data-handling, rollout-zarr, thesis]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_target_selection.py
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/aria_nbv/pose_generation/rollout_trace.py
  - aria_nbv/tests/data_handling/test_target_selection.py
  - aria_nbv/tests/data_handling/test_rollout_zarr_store.py
  - docs/contents/impl/rollout_storage_contract.qmd
  - docs/contents/impl/data_pipeline_overview.qmd
---

Implemented the first V1 actor-visible target selector as a data-handling
config-as-factory surface. The selector ranks observed/predicted OBB rows by
confidence, projected area, semidense/EVL support, and support deficit, then
returns deterministic `greedy_top_k` or seeded `temperature_softmax_top_k`
rows. V1 refuses GT-only inputs; GT is limited to V0 sanity selection or
post-selection matching for label eligibility.

Rollout lineage and `rollouts.zarr` target tables now preserve selector policy,
rank, score, probability/temperature, target invalidity bits, and GT match
metadata. Docs and canonical decisions were updated to make this selector the
first thesis-grade automatic target-selection contract.

Verification run during the pass:

- `cd aria_nbv && uv run pytest tests/data_handling/test_target_selection.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_target_selection.py aria_nbv/data_handling/__init__.py aria_nbv/pose_generation/rollout_trace.py aria_nbv/data_handling/_rollout_zarr_store.py tests/data_handling/test_target_selection.py tests/data_handling/test_rollout_zarr_store.py`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/impl/rollout_storage_contract.qmd`
- `cd docs && quarto render contents/impl/data_pipeline_overview.qmd`
- `make check-agent-memory`

---
id: 2026-03-30_aria_nbv_dedup_refactor
date: 2026-03-30
title: "Aria-NBV Dedup Refactor"
status: done
topics: [aria-nbv, vin, data-handling, pose-generation, refactor]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/vin/_plotting_common.py
  - aria_nbv/aria_nbv/vin/_model_mixins.py
  - aria_nbv/aria_nbv/vin/semidense_projection.py
  - aria_nbv/aria_nbv/vin/experimental/scorer_head.py
  - aria_nbv/aria_nbv/utils/reporting.py
  - aria_nbv/aria_nbv/data_handling/_config_utils.py
  - aria_nbv/aria_nbv/data_handling/_sample_keys.py
  - aria_nbv/aria_nbv/data_handling/_legacy_dataset_mixins.py
artifacts:
  - .agents/tmp/audit/out/post_refactor_v2.md
  - .agents/tmp/audit/out/post_refactor_v2.json
---

Task
- Resolve the duplicate helper/model/config logic highlighted by the repository redundancy audit for `aria_nbv`.

Method
- Extracted canonical shared owners for repeated VIN plotting, scorer-head, semidense projection, config, reporting, summary, cache-path, and token-sanitization helpers.
- Replaced repeated VIN model wrapper methods with shared mixins instead of keeping multiple identical delegating method bodies.
- Consolidated repeated pose-generation seed and pose-shape helpers into shared utilities.
- Re-ran the duplicate audit after refactoring to confirm the remaining hits were audit heuristics or legacy-surface debt rather than active code duplication.

Findings
- The meaningful duplicate helper and validator groups were reduced to zero in the audit.
- Remaining audit items are non-logic categories: `app/__init__.py` vs `data_handling/__init__.py` lazy-import symmetry, namespace-overlap heuristics for `pose_generation`/`rri_metrics`, helper-sprawl warnings, and legacy/dead-file candidates.
- The previous extraction had left one broken import in `vin/experimental/plotting.py`; it now imports the frustum-point helper from `vin_utils.py`, which is the canonical owner.

Verification
- `cd aria_nbv && .venv/bin/ruff check --ignore N999 ...touched files...`
- `cd aria_nbv && .venv/bin/ruff format ...touched files...`
- `cd aria_nbv && .venv/bin/python -m pytest --capture=no tests/vin/test_vin_plotting_v3.py tests/test_plotting_helpers_refactor.py tests/utils/test_plotting_utils.py`
- `cd aria_nbv && .venv/bin/python -m pytest --capture=no tests/pose_generation/test_align_to_gravity.py tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && .venv/bin/python -m pytest --capture=no tests/data_handling/test_public_api_contract.py tests/lightning/test_vin_batch_collate.py`
- `./aria_nbv/.venv/bin/python .agents/tmp/audit/repo_duplicate_audit.py --root aria_nbv/aria_nbv --out .agents/tmp/audit/out/post_refactor_v2.md --json-out .agents/tmp/audit/out/post_refactor_v2.json`

Residual risk
- One VIN-core test outside the deduped surfaces still fails: `tests/vin/test_vin_model_v3_methods.py::test_vin_oracle_batch_shuffle_candidates_batched`. The failing logic is in `aria_nbv/data_handling/vin_oracle_types.py`, which was not changed in this refactor.

Canonical state impact
- None. This was an internal code-structure cleanup with compatibility preserved.

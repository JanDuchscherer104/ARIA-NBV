---
id: 2026-03-30_mojo_vin_semidense_backend
date: 2026-03-30
title: "Mojo VIN Semidense Backend"
status: done
topics: [mojo, vin, semidense, benchmarking]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .agents/memory/state/PROJECT_STATE.md
  - aria_nbv/aria_nbv/vin/model_v3.py
  - aria_nbv/aria_nbv/vin/mojo_backend.py
  - aria_nbv/aria_nbv/vin/mojo/vin_projection_kernels.mojo
  - aria_nbv/tests/vin/test_vin_model_v3_methods.py
  - aria_nbv/scripts/benchmark_mojo_vin_projection.py
artifacts:
  - /tmp/nbv-mojo-vin-benchmark.json
---

Task: add another optional Mojo backend for a real `aria_nbv` bottleneck without replacing the current implementation, then benchmark it against the Torch path.

Method: kept `_project_semidense_points()` on the existing PyTorch3D path and factored the reduction-heavy semidense projection accumulation into a shared seam. Added `SemidenseProjectionBackend` to `VinModelV3Config`, a Python-importable Mojo reducer under `aria_nbv/aria_nbv/vin/mojo/`, caching so projection stats and grid CNN share one accumulation per forward pass, targeted Torch-vs-Mojo equivalence tests, and a dedicated benchmark script.

Findings: the new optional CPU Mojo backend accelerates both `_encode_semidense_projection_features()` and `_encode_semidense_grid_features()` while preserving the Torch path as default. On the synthetic CPU benchmark (`batch_size=2`, `num_candidates=48`, `num_points=4096`, `grid_size=24`, `repeats=7`), measured speedups were `2.40x` for raw accumulation, `2.96x` for scalar projection features, `2.26x` for grid features, and `2.19x` for the combined projection-stats-plus-grid path.

Verification:
- `ruff format` and `ruff check` on the touched VIN/backend/benchmark/test files
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python -m pytest -s -q ...`
  - direct semidense projection slice: `10 passed`
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python /tmp/nbv-mojo-skill-pr/aria_nbv/scripts/benchmark_mojo_vin_projection.py --case all --repeats 7 --batch-size 2 --num-candidates 48 --num-points 4096 --grid-size 24 --json-out /tmp/nbv-mojo-vin-benchmark.json`

Canonical state impact: `PROJECT_STATE.md` now records that experimental Mojo acceleration also covers VIN semidense projection accumulation, with Torch/PyTorch3D remaining the default path and Mojo still requiring an explicit config selector plus reachable site-packages.

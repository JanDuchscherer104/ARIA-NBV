---
id: 2026-04-14_mojo_oracle_backend
date: 2026-04-14
title: "Apple-Silicon Mojo Oracle Backend Scaffold"
status: done
topics: [mojo, oracle-rri, rendering, pose-generation, docs, tests]
confidence: medium
canonical_updates_needed: []
files_touched:
  - REQUIREMENTS.md
  - .agents/skills/mojo-nbv-acceleration/SKILL.md
  - .agents/skills/mojo-nbv-acceleration/references/mojo-context7-summary.md
  - .agents/memory/state/PROJECT_STATE.md
  - docs/contents/impl/apple_silicon_mojo_oracle_backend.qmd
  - docs/contents/impl/oracle_rri_impl.qmd
  - docs/index.qmd
  - docs/typst/paper/sections/05-oracle-rri.typ
  - aria_nbv/aria_nbv/pose_generation/types.py
  - aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py
  - aria_nbv/aria_nbv/pose_generation/mojo_backend.py
  - aria_nbv/aria_nbv/pose_generation/mojo/mesh_collision_kernels.mojo
  - aria_nbv/aria_nbv/rendering/candidate_depth_renderer.py
  - aria_nbv/aria_nbv/rendering/candidate_pointclouds.py
  - aria_nbv/aria_nbv/rendering/mojo_backend.py
  - aria_nbv/aria_nbv/rendering/mojo_depth_renderer.py
  - aria_nbv/aria_nbv/rendering/mojo/oracle_render_kernels.mojo
  - aria_nbv/aria_nbv/rri_metrics/oracle_rri.py
  - aria_nbv/aria_nbv/rri_metrics/mojo_backend.py
  - aria_nbv/aria_nbv/rri_metrics/mojo/oracle_distance_kernels.mojo
  - aria_nbv/tests/pose_generation/test_mojo_collision_backend.py
  - aria_nbv/tests/rendering/test_oracle_backend_contracts.py
  - aria_nbv/tests/rri_metrics/test_mojo_oracle_backend.py
  - aria_nbv/tests/integration/test_oracle_rri_backend_parity.py
  - aria_nbv/scripts/benchmark_mojo_candidate_generation.py
  - aria_nbv/scripts/benchmark_mojo_oracle_rri.py
artifacts:
  - /tmp/nbv-mojo-diagram-1.svg
  - /tmp/nbv-mojo-diagram-2.svg
---

Task: implement the planned Apple-Silicon Mojo backend surface for the oracle
RRI pipeline, with docs and tests written first.

Method: added strict requirements docs, a new architecture/design page with
validated Mermaid diagrams, vendored the Mojo acceleration skill scaffold,
wired backend-selection enums and configs across candidate generation,
rendering, point-cloud construction, and oracle distance scoring, and added
Python-importable Mojo kernel modules for collision, depth rendering,
backprojection, and point↔mesh distance work. Added parity-focused tests and
benchmark entrypoints.

Findings: the Mojo extension modules compile and import successfully through the
repo-local `.mojo-venv` on this Apple-Silicon machine. Direct kernel smoke
checks passed for collision, rendering, and distance modules. The broader repo
pytest stack remains blocked in this workspace because the expected vendored
`external/efm3d` Python project is missing, so package-level integration tests
could not be executed end-to-end here.

Verification:
- `python3 -m py_compile ...` on the touched Python files
- direct Mojo importer checks for:
  - `mesh_collision_kernels`
  - `oracle_render_kernels`
  - `oracle_distance_kernels`
- direct `.mojo-venv` kernel smoke script covering collision, depth, and
  distance modules
- `ruff check` on the touched Python files
- Mermaid validation with `npx -y @mermaid-js/mermaid-cli`
- `make check-agent-memory`

---
id: 2026-03-30_mojo_collision_backend
date: 2026-03-30
title: "Mojo Collision Backend"
status: done
topics: [mojo, pose-generation, benchmarking, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/pose_generation/mojo_backend.py
  - aria_nbv/aria_nbv/pose_generation/mojo/mesh_collision_kernels.mojo
  - aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py
  - aria_nbv/aria_nbv/pose_generation/types.py
  - aria_nbv/scripts/benchmark_mojo_candidate_generation.py
  - aria_nbv/tests/pose_generation/test_pose_generation.py
  - .agents/memory/state/PROJECT_STATE.md
artifacts:
  - aria_nbv/scripts/benchmark_mojo_candidate_generation.py
---

Task: Add a switchable Mojo-backed candidate-generation path without replacing the current implementations, verify equivalence against the Trimesh behavior, and benchmark the result.

Method: Installed Mojo in a repo-local `.mojo-venv`, validated Python import and raw-pointer buffer access, added a lazy Python bridge plus a Mojo kernel module for point-to-mesh distance and clearance masks, extended `CollisionBackend` with `MOJO`, and routed `MinDistanceToMeshRule` through the new path. Path-collision checks were initially prototyped in Mojo as well, but dense-mesh generator runs diverged from Trimesh, so the shipped `MOJO` backend keeps the existing Trimesh ray engine for `PathCollisionRule` and only accelerates the verified mesh-clearance stage.

Findings: The new backend is config-selectable, UI-selectable, and preserves existing code paths. Targeted equivalence tests passed for the direct clearance and path-collision rules. Benchmarking on a synthetic 1,280-face icosphere with `num_samples=256` showed an isolated mesh-clearance speedup of about `8.96x` (`141.413 ms` Trimesh vs `15.791 ms` Mojo with `ensure_collision_free=False`) and an end-to-end candidate-generation speedup of about `1.30x` (`163.826 ms` vs `126.261 ms`) when the unchanged path-collision rule remained enabled.

Verification:
- `ruff check aria_nbv/aria_nbv/pose_generation/mojo_backend.py aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py aria_nbv/aria_nbv/pose_generation/types.py aria_nbv/aria_nbv/app/ui.py aria_nbv/tests/test_pose_generation.py aria_nbv/tests/pose_generation/test_pose_generation.py aria_nbv/scripts/benchmark_mojo_candidate_generation.py`
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python -m pytest -s /tmp/nbv-mojo-skill-pr/aria_nbv/tests/pose_generation/test_pose_generation.py /tmp/nbv-mojo-skill-pr/aria_nbv/tests/test_pose_generation.py -q`
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python /tmp/nbv-mojo-skill-pr/aria_nbv/scripts/benchmark_mojo_candidate_generation.py --num-samples 256 --mesh-subdivisions 3 --repeats 5`
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python /tmp/nbv-mojo-skill-pr/aria_nbv/scripts/benchmark_mojo_candidate_generation.py --num-samples 256 --mesh-subdivisions 3 --repeats 5 --ensure-collision-free`

Canonical state impact: `PROJECT_STATE.md` now records the experimental `CollisionBackend.MOJO` path and the requirement that Mojo site-packages be reachable through the repo-local install or `ARIA_NBV_MOJO_SITE_PACKAGES`.

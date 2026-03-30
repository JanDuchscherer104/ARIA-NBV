---
id: project_state
updated: 2026-03-30
scope: repo
owner: jan
status: active
tags: [nbv, rri, efm3d, ase, codex]
---

# Project State

## Goal and Claims
This repository develops an active next-best-view planner for egocentric indoor scenes. The core claim is that ranking candidate viewpoints by relative reconstruction improvement can outperform proxy objectives such as pure coverage, especially when combined with frozen egocentric foundation-model features from EFM3D or EVL.

## Current Architecture
The current stack has three main layers: ASE and EFM-facing data access, oracle label generation for candidate views, and VIN-style learned scoring on top of frozen backbone features. Candidate viewpoints are sampled around the reference trajectory pose, rendered against ground-truth meshes, fused with semi-dense SLAM points, and scored with RRI-derived labels. Training and diagnostics live in the `aria_nbv` package under `aria_nbv/aria_nbv`, while Quarto and Typst document theory, implementation, and experiments.

## Stable Conventions
- Treat `docs/typst/paper/main.typ` as the highest-level project narrative and sync Quarto docs to it.
- Use the uv-managed environment in `aria_nbv/.venv`.
- Use `PoseTW` and `CameraTW` instead of raw matrices.
- Treat `docs/references.bib` as the only bibliography source of truth.
- Keep repo guidance in `AGENTS.md`, repeatable workflows in `.agents/skills/`, and generated context in `docs/_generated/context/`.
- Keep operator aids and long-form conventions in `.agents/references/`; canonical state docs should remain focused on current truth.
- The default Codex bootstrap is `docs/typst/paper/main.typ` + `.agents/memory/state/` + the compact `docs/_generated/context/source_index.md`, with broader references retrieved on demand.
- Treat `make context-contracts` / `scripts/nbv_get_context.sh contracts` as the preferred contract surface; heavy generated artifacts are fallback-only.
- Treat `aria_nbv.data_handling` as the canonical owner of raw snippet, oracle-cache, VIN-cache, and cache-coverage contracts, and `aria_nbv.utils.data_plotting` as the canonical owner of shared snippet plotting; mirrored `aria_nbv.data` compatibility modules were removed.
- Remaining legacy oracle-cache / VIN-snippet-cache runtime, UI, CLI, and dedicated test surfaces are tagged with `NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION` so the final removal sweep can be done via one grep query.
- The `aria_nbv.data_handling` package root is now canonical-only; remaining legacy cache APIs live behind dedicated `_legacy_cache_api.py` and `_legacy_vin_source.py` modules, while the old public submodule names (`oracle_cache.py`, `vin_cache.py`, `vin_oracle_datasets.py`, etc.) are thin compatibility wrappers over those `_legacy_*` owners.

## Active Experiments
The project is actively iterating on VIN variants, semidense projection cues, candidate generation behavior, and documentation alignment between code, paper, and slides. Candidate generation now also has an experimental `CollisionBackend.MOJO` path for mesh-clearance queries; path-collision checks still delegate to the established Trimesh ray engine for equivalence while the parallel Mojo distance kernel is evaluated.

## Risks and Pitfalls
- Validation can be disabled by config defaults if Lightning is misconfigured.
- Interpreter mismatch can break tests when `uv run` or the repo venv is not used.
- Candidate-pose frame consistency and CW90 corrections remain easy to misuse across rendering and VIN inputs.
- The experimental Mojo backend requires the Mojo Python package to be reachable, either from `<repo>/.mojo-venv` or via `ARIA_NBV_MOJO_SITE_PACKAGES`.

## Pointers
- `docs/index.qmd`
- `docs/contents/todos.qmd`
- `docs/contents/impl/`
- `docs/contents/ext-impl/`

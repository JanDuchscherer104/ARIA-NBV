---
id: project_state
updated: 2026-03-24
scope: repo
owner: jan
status: active
tags: [nbv, rri, efm3d, ase, codex]
---

# Project State

## Goal and Claims
This repository develops an active next-best-view planner for egocentric indoor scenes. The core claim is that ranking candidate viewpoints by relative reconstruction improvement can outperform proxy objectives such as pure coverage, especially when combined with frozen egocentric foundation-model features from EFM3D or EVL.

## Current Architecture
The current stack has three main layers: ASE and EFM-facing data access, oracle label generation for candidate views, and VIN-style learned scoring on top of frozen backbone features. Candidate viewpoints are sampled around the reference trajectory pose, rendered against ground-truth meshes, fused with semi-dense SLAM points, and scored with RRI-derived labels. Training and diagnostics live in the `oracle_rri` package, while Quarto and Typst document theory, implementation, and experiments.

## Stable Conventions
- Treat `docs/typst/paper/main.typ` as the highest-level project narrative and sync Quarto docs to it.
- Use the uv-managed environment in `oracle_rri/.venv`.
- Use `PoseTW` and `CameraTW` instead of raw matrices.
- Treat `docs/references.bib` as the only bibliography source of truth.
- Keep repo guidance in `AGENTS.md`, repeatable workflows in `.agents/skills/`, and generated context in `docs/_generated/context/`.

## Active Experiments
The project is actively iterating on VIN variants, semidense projection cues, candidate generation behavior, and documentation alignment between code, paper, and slides.

## Risks and Pitfalls
- Validation can be disabled by config defaults if Lightning is misconfigured.
- Interpreter mismatch can break tests when `uv run` or the repo venv is not used.
- Candidate-pose frame consistency and CW90 corrections remain easy to misuse across rendering and VIN inputs.

## Pointers
- `docs/index.qmd`
- `docs/contents/todos.qmd`
- `docs/contents/impl/`
- `docs/contents/ext-impl/`

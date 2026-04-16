---
scope: package
applies_to: aria_nbv/**
summary: Package-wide implementation, validation, and routing guidance for the Aria-NBV Python workspace.
---

# Package Guidance

This file applies to work under `aria_nbv/` and adds package-specific deltas on
top of the root [AGENTS.md](../AGENTS.md). Read the nearest nested `AGENTS.md`
before editing a localized subtree.

## Local Sources Of Truth
- [README.md](README.md): package overview and developer workflow.
- [../.agents/references/python_conventions.md](../.agents/references/python_conventions.md):
  long-form typing, docstring, tensor-shape, and config examples.
- [../.agents/memory/state/GOTCHAS.md](../.agents/memory/state/GOTCHAS.md):
  maintained package pitfalls and verification traps.
- The nearest package README or nested `AGENTS.md` for local ownership notes.

## Core Rules
- Use `pathlib.Path` for filesystem paths.
- Use `PoseTW` and `CameraTW` instead of raw matrices at normal package
  boundaries.
- Use `Console` from `aria_nbv.utils` for structured logging.
- Prefer existing implementations from PyTorch, PyTorch3D, EFM3D, ATEK, and
  Project Aria tools over local reimplementation.
- Use Aria constants from `efm3d.aria.aria_constants` for dataset keys.
- Document tensor shapes, coordinate frames, transform directionality, and
  candidate-frame assumptions where they are not obvious.
- Never let package behavior fail silently; raise actionable errors or log
  explicit failure context.

## Config And Contracts
- Config classes should inherit `BaseConfig` where appropriate and remain the
  main construction surface for runtime objects.
- Instantiate runtime objects through config `.setup_target()` methods rather
  than loose dicts or long argument lists.
- Prefer `Field(default_factory=...)` for computed defaults and nested config
  defaults.
- Use `field_validator`, `model_validator`, and `setup_target()` together for
  validation, default wiring, and runtime instantiation.
- Keep package-boundary containers typed; avoid ad hoc `dict[str, Any]` payloads
  when a dedicated config, dataclass, or model should exist.

## Deeper Guides
- `aria_nbv/aria_nbv/app/AGENTS.md`: Streamlit UI, panels, state, and plotting
  composition.
- `aria_nbv/aria_nbv/configs/AGENTS.md`: path, W&B, Optuna, and persisted config
  ownership.
- `aria_nbv/aria_nbv/data_handling/AGENTS.md`: raw snippets, cache stores,
  splits, and VIN cache contracts.
- `aria_nbv/aria_nbv/lightning/AGENTS.md`: experiment, trainer, datamodule, and
  training-loop contracts.
- `aria_nbv/aria_nbv/pipelines/AGENTS.md`: pipeline entrypoints and oracle RRI
  labeler orchestration.
- `aria_nbv/aria_nbv/pose_generation/AGENTS.md`: candidate generation,
  feasibility rules, and counterfactual pose helpers.
- `aria_nbv/aria_nbv/rendering/AGENTS.md`: depth rendering, unprojection,
  point-cloud construction, and diagnostics.
- `aria_nbv/aria_nbv/rl/AGENTS.md`: discrete-shell RL and counterfactual
  environment contracts.
- `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`: oracle RRI, binning, ordinal loss,
  and reported metric semantics.
- `aria_nbv/aria_nbv/vin/AGENTS.md`: scorer inputs, candidate context, shared
  batch containers, and VIN-facing frame semantics.

## Verification
- For package changes, run format, lint, and targeted pytest on the changed
  surface.
- Every new feature or behavior change should come with targeted pytest
  coverage unless the work is documentation-only.
- Prefer real-data or integration-style tests when feasible.
- Update docs when behavior or user-facing workflows change.
- Keep public signatures typed and relevant public methods documented.

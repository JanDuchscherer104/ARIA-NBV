---
scope: package
applies_to: aria_nbv/**
summary: Package-specific implementation, validation, and design guidance for the aria_nbv Python workspace.
---

# Package Guidance

Apply this file when working under `aria_nbv/`.

## Commands
- Python: `aria_nbv/.venv/bin/python`
- Environment recovery: `cd aria_nbv && uv sync --all-extras`
- Format: `ruff format <file>`
- Lint: `ruff check <file>`
- Tests: `uv run pytest <path>` or `aria_nbv/.venv/bin/python -m pytest <path>`
- Context: `make context`
- Contracts: `make context-contracts`

## Core Rules
- Use `pathlib.Path` for filesystem paths.
- Use `PoseTW` and `CameraTW` instead of raw matrices.
- Use `Console` from `aria_nbv.utils` for structured logging.
- Prefer existing implementations (i.e. from `pytorch3d`, `efm3d`, `atek`, and `projectaria_tools`) over reimplementation.
- Use ARIA constants from `efm3d.aria.aria_constants` for dataset keys.
- Follow EFM3D / ATEK coordinate conventions and document tensor shapes plus coordinate frames where they are not obvious.
- Never let package behavior fail silently; raise actionable errors or log explicit failure context.

## Ownership Boundaries
- New raw-snippet or cache pipeline work should target `aria_nbv.data_handling`; treat `aria_nbv.data` as the compatibility surface and only extend it when the task explicitly requires backward compatibility.
- Once the task localizes to scorer, cache/data, or oracle-metric internals, open the deeper guide in `aria_nbv/aria_nbv/vin/AGENTS.md`, `aria_nbv/aria_nbv/data_handling/AGENTS.md`, or `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`.

## Config-As-Factory
- Config classes should inherit `BaseConfig` and remain the main construction surface for runtime objects.
- Instantiate runtime objects through config `.setup_target()` methods rather than loose dicts or long argument lists.
- Nested configs should compose subcomponents when that improves clarity; do not bypass nested configs that already exist.
- Use `setup_target(...)` for late-bound runtime inputs such as `params`, `trainer`, or `split`.
- Prefer `Field(default_factory=...)` for computed defaults and nested config defaults.
- Use `field_validator`, `model_validator`, and `setup_target()` together for validation, default wiring, and runtime instantiation logic.
- Canonical examples: `aria_nbv/aria_nbv/utils/base_config.py` and `aria_nbv/aria_nbv/lightning/aria_nbv_experiment.py`.

## Anti-Patterns
- Do not instantiate internal runtime objects from raw `dict[str, Any]` blobs when a dedicated config model should exist.
- Do not bypass a nested config object to construct one of its targets manually.
- Do not use `Field(..., description=...)` as the primary documentation for config fields; prefer attribute or field docstrings.
- Do not add compatibility wrappers or silent fallbacks when removing obsolete internal interfaces unless the task explicitly asks for compatibility.

## Code Quality
- All interfaces must be fully typed and use modern builtins such as `list[str]` and `dict[str, Any]`.
- Use `TYPE_CHECKING` guards for type-only imports.
- Use `Literal` for constrained string values.
- Public methods, functions, classes and modules must have Google-style docstrings. Each module doc-string must give a high-level overview of the module's purpose and contents.
- When shapes, coordinate frames, or transform directionality are non-obvious, document them explicitly in code and docstrings.
- See [python_conventions.md](../.agents/references/python_conventions.md) for full examples and anti-patterns.


## Verification
- For package changes, run format -> lint -> targeted pytest on the changed surface.
- Every new feature or behavior change should come with targeted pytest coverage.
- Prefer real-data or integration-style tests when feasible.
- Update docs when behavior or user-facing workflows change.
- Keep public signatures typed and public methods documented.

## Completion Criteria
- Changed Python files are formatted and lint-clean.
- Targeted pytest coverage was run for the changed surface.
- Docs were updated when behavior or user-facing workflows changed.
- No temporary placeholders, stale paths, or undocumented public API changes remain.

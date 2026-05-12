---
id: 2026-05-11_typed_config_targets
date: 2026-05-11
title: "Typed Config Targets"
status: done
topics: [python, configs, typing]
confidence: high
canonical_updates_needed: []
---

## Task

Implemented typed config-as-factory return contracts for `setup_target()` on
2026-05-11. The goal was to keep target-less `BaseConfig` models compatible
while giving target-producing configs precise static return types.

## Method

Added `TargetConfig[T]` above the existing `BaseConfig` factory helper and
migrated target-producing package configs to inherit `TargetConfig[...]`.
Factory declarations now use the canonical `target_type` property; `target`
remains as a compatibility alias for external or unmigrated callers.

## Verification

- `uv run ruff check` on the touched Python files.
- `uv run pytest tests/test_base_config_toml.py`
- `uv run pytest tests/configs/test_wandb_config_resume.py`
- `uv run mypy aria_nbv/utils/base_config.py tests/test_base_config_toml.py`
- Targeted `mypy -c` reveal checks for `Pytorch3DDepthRendererConfig` and
  `AdamWConfig` confirmed typed `setup_target()` return values.

## Canonical State Impact

No canonical project-state update is required. This is a shared typing and
config API cleanup with no intended runtime behavior change.

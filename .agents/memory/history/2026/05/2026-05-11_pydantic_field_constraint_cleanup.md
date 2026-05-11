---
id: 2026-05-11_pydantic_field_constraint_cleanup
date: 2026-05-11
title: "Pydantic Field Constraint Cleanup"
status: done
topics: [aria-nbv, python, config, validation]
confidence: high
canonical_updates_needed: []
---

## Task

Replaced package-local manual `field_validator` methods that only enforced
simple numeric/list/tuple bounds with declarative Pydantic `Field` constraints.

## Method

Kept validators that perform path resolution, device or verbosity coercion,
CLI/string normalization, crop-policy guards, and model-compatibility checks.
Converted scalar bounds, seed bounds, list item bounds, even Fourier dimensions,
and positive pose-scale/depth constraints where they matched existing behavior.

## Verification

- `cd aria_nbv && uv run pytest tests/test_config_field_constraints.py -q`
- `cd aria_nbv && uv run pytest tests/test_config_field_constraints.py tests/data_handling/test_target_selection.py tests/pose_generation tests/rerun_inspector -q`
- `cd aria_nbv && uv run pytest tests/data_handling tests/pose_generation tests/rerun_inspector tests/app tests/vin`
- `cd aria_nbv && uv run ruff check <touched files>`
- `git diff --check`

## Notes

The broad package test target initially failed during collection because
`aria_nbv.lightning.aria_nbv_experiment` referenced `pl` at runtime in a generic
base class while importing it only under `TYPE_CHECKING`. A runtime-safe target
type alias was added so the requested app and VIN tests could collect.

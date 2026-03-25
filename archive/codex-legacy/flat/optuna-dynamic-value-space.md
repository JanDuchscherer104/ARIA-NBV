# Optuna: categorical search space changes in an existing study

Date: 2026-01-07

## Symptom

When running `nbv-optuna` on an existing study (e.g. `.logs/optuna/vin-v2-sweep.db`), Optuna can fail early with:

```
ValueError: CategoricalDistribution does not support dynamic value space.
```

This happens when a parameter **keeps the same name** (we use the dotted config path, e.g.
`module_config.vin.use_point_encoder`) but its categorical `choices=(...)` set is changed (e.g. `(True, False)` →
`(True,)`).

## Root cause

Optuna enforces **distribution compatibility per parameter name** within a single study. For categorical parameters, the
choice set is part of the distribution, so shrinking/expanding it under the same key is rejected.

## Fix implemented

`oracle_rri/utils/optuna_optimizable.py::Optimizable.suggest` now treats **single-choice categorical** parameters as
**fixed**:

- If `len(choices) == 1`, it returns that value directly and **does not call** `trial.suggest_categorical(...)`.
- If `len(choices) == 0`, it raises a clear `ValueError`.

This avoids Optuna’s dynamic-value-space error when “freezing” a categorical parameter mid-study by setting its choice
space to a singleton.

## Notes / alternatives

- With the singleton-as-fixed behavior, the parameter may no longer appear in `trial.params` for new trials (because we
  don't register it as an Optuna parameter). This is usually fine since the value is constant.
- If you need the fixed value to be stored in Optuna params, the most robust approach is to start a **new study**
  (new `study_name`) when changing categorical choice sets, or to **rename** the parameter key (e.g. via
  `Optimizable(..., name="...")`) so Optuna treats it as a new dimension.


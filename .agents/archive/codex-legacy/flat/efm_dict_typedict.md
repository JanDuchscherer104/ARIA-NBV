# Task: Define `EfmDict` for VIN/EVL outputs

## Summary
- Implemented `oracle_rri/oracle_rri/vin/types.py` `EfmDict` as a `TypedDict` covering the EVL output keys used by VIN-style workflows.
- Added per-field shape/meaning docs via `typing_extensions.Doc` + `typing.Annotated` (TypedDict fields cannot carry standalone docstrings like dataclasses).
- Fixed a parsing error in `types.py` that previously broke `make context`.

## Validation
- `make context` now succeeds.
- Ran:
  - `ruff format oracle_rri/oracle_rri/vin/types.py`
  - `ruff check oracle_rri/oracle_rri/vin/types.py`
  - `pytest -q oracle_rri/tests/vin/test_types.py`

## Notes / follow-ups
- Full test suite currently fails during collection due to unrelated missing imports:
  - `oracle_rri.pose_generation.reference_power_spherical_distributions`
  - `oracle_rri.rendering.plotting.hit_ratio_bar`
  - `oracle_rri.data.mesh_cache.mesh_from_snippet`
  - `oracle_rri.data.efm_dataset.crop_mesh_with_bounds` / `infer_semidense_bounds`
  - `oracle_rri.utils.performance`
- `EvlBackboneOutput` docstring describes several fields as “Optional”, but the type annotations are currently non-optional (`Tensor` not `Tensor | None`). If you want stricter typing, consider aligning the annotations with the actual runtime behavior of `EvlBackbone.forward(...)` under different `features_mode`s.


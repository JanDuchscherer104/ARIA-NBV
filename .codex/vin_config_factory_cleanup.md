# VIN config-as-factory cleanup (2025-12-19)

## Goal

Enforce the project’s config-as-factory pattern across VIN so that:

- `BaseConfig.setup_target()` always passes the *config object itself* into runtime targets.
- Redundant runtime `int(...)`/`bool(...)` casts and “config value” checks are removed.
- Numeric constraints live in Pydantic fields/validators (fail fast at config construction).

## What changed

- `oracle_rri/oracle_rri/vin/pose_encoding.py`
  - `FourierFeatures` + `LearnableFourierFeatures` now take `config: *Config` in `__init__`.
  - Moved parameter validation into `FourierFeaturesConfig` / `LearnableFourierFeaturesConfig` via `Field(...)` constraints + `fourier_dim` even validator.
  - Removed custom `setup_target()` overrides that passed raw parameters.

- `oracle_rri/oracle_rri/vin/spherical_encoding.py`
  - `ShellShPoseEncoder` now takes `config: ShellShPoseEncoderConfig` in `__init__`.
  - Added Pydantic constraints (`Field(gt=0/ ge=0)` etc.) to `ShellShPoseEncoderConfig`.
  - Removed custom `setup_target()` override that passed raw parameters.
  - Radius Fourier features are built via an internal `FourierFeaturesConfig(...).setup_target()` (keeps implementation reusable without changing the external config surface).

- `oracle_rri/oracle_rri/vin/model.py`
  - `VinScorerHead` now takes `config: VinScorerHeadConfig` in `__init__` (+ optional `in_dim`).
  - `VinScorerHeadConfig.setup_target(in_dim=...)` now instantiates via `self.target(self, in_dim=...)`.
  - Removed a leftover `int(...)` cast in `_largest_divisor_leq`.

- `oracle_rri/oracle_rri/vin/backbone_evl.py`
  - Removed redundant `str(self.config.features_mode)` cast (it’s already a `Literal[...]`).

- `oracle_rri/tests/vin/test_pose_encoding.py`
  - Updated to instantiate `LearnableFourierFeatures` via `LearnableFourierFeaturesConfig(...).setup_target()`.

## Verification

- `ruff format` + `ruff check` on touched VIN files passed.
- `pytest` (subset) passed:
  - `oracle_rri/tests/vin/test_pose_encoding.py`
  - `oracle_rri/tests/vin/test_rri_binning.py`
  - `tests/vin/test_frustum_unproject_p3d.py`
  - `tests/vin/test_candidate_validity.py`
  - `tests/vin/test_rri_binning.py`
  - `tests/vin/test_vin_model_integration.py::test_vin_forward_on_real_snippet_cpu`

## Notes / potential follow-ups

- There are *two* VIN test locations (`oracle_rri/tests/vin` and `tests/vin`). It works, but it’s easy to run the wrong path; consider consolidating to one test root.
- `FourierFeatures` / `LearnableFourierFeatures` constructors are now config-only. If you still want ergonomic direct construction in notebooks, consider adding a `@classmethod from_params(...)` helper (kept out-of-scope for this patch).
- `ShellShPoseEncoder` currently constructs an internal `FourierFeaturesConfig` at runtime. This is fine (once per model init), but if you want “pure declarative configs”, consider making it an explicit nested field on `ShellShPoseEncoderConfig` (would change config schema).


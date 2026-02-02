# Learnable Fourier Features (LFF) review — 2026-01-07

## Context

We reviewed `oracle_rri/oracle_rri/vin/pose_encoding.py::LearnableFourierFeatures` against the commonly referenced
PyTorch implementation in `JHLew/Learnable-Fourier-Features` (`positional_encoding.py`).

## Findings

- Core forward path matches the reference for the `g_dim=1` case:
  - `XWr = pos @ Wr.T`
  - `F = [cos(XWr), sin(XWr)] / sqrt(f_dim)`
  - `Y = MLP(F)` → positional embedding
- Our implementation intentionally does **not** implement the reference’s optional grouping dimension `g_dim`
  and the final `rearrange(Y, 'b l g d -> b l (g d)')`. For VIN, pose vectors are passed in as flat `[..., D]`
  so this was not needed.
- The main discrepancy was the **initialization scaling** for `Wr`:
  - Reference: `Wr = randn(...) * (gamma ** 2)`
  - Ours (before fix): `Wr = randn(...) * gamma`

## Changes made

- Updated `LearnableFourierFeatures.Wr` init to use `gamma ** 2` to match the reference implementation.
  - File: `oracle_rri/oracle_rri/vin/pose_encoding.py`
- Added unit tests to lock in the reference init behavior and document `include_input=True` semantics.
  - File: `tests/vin/test_learnable_fourier_features.py`

## Notes / open suggestions

- `include_input=True` uses `torch.cat([x, enc], dim=-1)` as an optional residual/identity channel; the reference
  implementation always returns only the learned encoding.
- `tests/vin/test_vin_model_v2_integration.py` currently fails because `VinPrediction` does not expose a
  `valid_frac` attribute (it has `semidense_candidate_vis_frac` / `semidense_valid_frac` instead). Consider adding a
  backward-compat alias (property or field) if `valid_frac` is still used by downstream code/tests.

## Local verification

- `oracle_rri/.venv/bin/ruff format oracle_rri/oracle_rri/vin/pose_encoding.py tests/vin/test_learnable_fourier_features.py`
- `oracle_rri/.venv/bin/ruff check oracle_rri/oracle_rri/vin/pose_encoding.py tests/vin/test_learnable_fourier_features.py`
- `oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_learnable_fourier_features.py`
- `oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_model_integration.py::test_vin_forward_on_real_snippet_cpu`


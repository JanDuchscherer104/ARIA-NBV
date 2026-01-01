# TOML serialization: `torch.device` values

- **Problem**: `BaseConfig.to_toml()` raised `tomlkit.exceptions.ConvertError` when a config contained a `torch.device` (e.g., `EvlBackboneConfig.device`) because tomlkit can't serialize `torch.device` objects directly.
- **Fix**: Normalize `torch.device` to a TOML string via `str(device)` during TOML serialization.
  - Change is in `oracle_rri/oracle_rri/utils/base_config.py` (`BaseConfig._normalise_scalar`).
- **Regression test**: Added `oracle_rri/tests/test_base_config_toml.py` to ensure `torch.device("cpu")` round-trips through TOML as `"cpu"`.
- **Validation**:
  - `ruff format --check --no-cache` and `ruff check --no-cache` on touched files.
  - `pytest oracle_rri/tests/test_base_config_toml.py` (passes).

## Notes / follow-ups

- If configs start carrying other Torch types (e.g., `torch.dtype`), consider extending the same normalization approach.
- Importing `oracle_rri` pulls in `wandb` via config modules; this creates temp dirs at import time. That’s fine in normal dev environments, but it makes minimal “import-only” checks harder in restricted sandboxes.


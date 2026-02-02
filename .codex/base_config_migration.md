## BaseConfig/SingletonConfig migration (2026-01-27)

### Summary
- Replaced `oracle_rri.utils.base_config.BaseConfig` with a hybrid implementation:
  - `target` is now a property (doc_classifier style).
  - TOML serialization/deserialization uses `tomli_w` + doc_classifier settings sources.
  - Retained oracle_rri JSON helpers (`model_dump_jsonable`, `model_dump_cache`, `to_jsonable`) and `_resolve_device`.
  - CLI keeps kebab-case support.
  - `propagated_fields` now stored as a `PrivateAttr` with a read-only property.
- Updated all config classes to override `target` as a property instead of a `Field`.
- Added `tomli_w` dependency to `oracle_rri/pyproject.toml`.

### Notes
- TOML output no longer includes comments/type hints (doc_classifier behavior). Kept conversion of torch.device/dtype/tensor in `_toml_normalize` to avoid serialization errors.
- `propagated_fields` is no longer part of `model_fields` (cleaner TOML/JSON output).

### Tests
- `ruff format` on all touched files.
- `ruff check` on touched files; only failure is pre-existing `N999` for `model_v1_SH.py` (invalid module name).
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/test_base_config_toml.py` failed: `tomli_w` not installed in the current venv (requires `uv sync` after dependency update).

### Follow-ups
- If desired, silence `N999` for `model_v1_SH.py` via per-file ignore or config.
- Consider whether TOML comment output is still needed anywhere (now removed).

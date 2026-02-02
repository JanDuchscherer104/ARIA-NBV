# VIN Diagnostics: vin_snippet_cache metadata IsADirectoryError (2026-01-29)

## Symptom
- Streamlit `VIN Diagnostics` panel crashes with:
  - `IsADirectoryError: [Errno 21] Is a directory: '/.../offline_cache/vin_snippet_cache'`
- Streamlit warns:
  - `The widget with key "vin_cache_index" was created with a default value but also had its value set via the Session State API.`

## Root cause
- `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py` passed the VIN snippet *cache directory* to
  `read_vin_snippet_cache_metadata(...)`, which previously expected a path to the JSON file itself.
- `Path.read_text()` on a directory triggers `IsADirectoryError`.
- The `vin_cache_index` widget both (a) had a `value=` default and (b) was updated via `st.session_state`,
  which Streamlit explicitly warns about.

## Fix
- `oracle_rri/oracle_rri/data/vin_snippet_cache.py`
  - `read_vin_snippet_cache_metadata()` now accepts either:
    - `.../vin_snippet_cache/metadata.json`, or
    - `.../vin_snippet_cache/` (auto-resolves to `metadata.json`).
- `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`
  - Pass `vin_cfg.cache.metadata_path` (explicit file path) when reading metadata.
  - Make the cache-index widget use session state for defaults (no `value=`), eliminating the warning.
- `tests/data/test_vin_snippet_cache.py`
  - Add a regression test ensuring directory-path metadata reads work.
  - Update the test metadata helper to include the now-required `pad_points` in the hash/payload.

## Verification
- `ruff format` + `ruff check` on touched files.
- `pytest -q tests/data/test_vin_snippet_cache.py` (all 5 tests pass).

## Follow-ups (optional)
- Consider displaying a friendlier UI error when `metadata.json` is missing/corrupt (instead of crashing the panel).


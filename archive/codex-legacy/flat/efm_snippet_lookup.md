# EFM snippet lookup

## Summary
- Added snippet-aware shard resolution and optional sample-key filtering in `oracle_rri/oracle_rri/data/efm_dataset.py`.
- Documented snippet selection in `oracle_rri/oracle_rri/data/README.md`.
- Added real-data integration test `tests/data/test_efm_dataset_snippet.py` covering sample-key lookup and shard-id lookup.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_efm_dataset_snippet.py -q`
  - Passes (uses real shards in `.data/ase_efm`).
- `pytest tests/data/test_efm_dataset_snippet.py -q` failed under system Python due to missing `power_spherical` (env mismatch).

## Notes / Suggestions
- `ruff check` on `oracle_rri/oracle_rri/data/efm_dataset.py` reports numerous pre-existing lint issues (imports, docstrings, complexity). Consider a cleanup pass if linting is enforced more strictly.
- If snippet IDs without scene IDs are expected (sample keys or shard names), consider enhancing `_autofill_paths` to infer scene IDs from sample-key prefixes before scanning shards.

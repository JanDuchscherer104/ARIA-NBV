# Task: Collapse offline cache serialization helpers

## Summary
- Collapsed the repetitive encode/decode wrappers in `oracle_rri/oracle_rri/data/offline_cache_serialization.py` into aliases/partials that still route through `encode_dataclass`/`decode_dataclass`.
- Kept the public API surface (`encode_*`, `decode_*`) intact while removing duplicated wrapper bodies.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/data/offline_cache_serialization.py`
- `uv run ruff check oracle_rri/oracle_rri/data/offline_cache_serialization.py`
- `uv run pytest tests/data/test_offline_cache.py -rs` (SKIPPED: missing `power_spherical` dependency)

## Open Items / Suggestions
- Install `power_spherical` (and confirm real data assets are present) to run `tests/data/test_offline_cache.py` without skipping.

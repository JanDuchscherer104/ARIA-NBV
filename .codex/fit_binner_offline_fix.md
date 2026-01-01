# fit_binner_offline val_cache ValidationError Fix

## Findings
- `VinDataModuleConfig.val_cache` was assigned a function via `Field(default=...)`, causing Pydantic to treat the function as a value and fail validation.

## Changes
- Reverted `val_cache` default to `None` so Pydantic doesn’t receive a function.

## Tests
- `pytest tests/data/test_offline_cache_split.py`

## Suggestions
- If you want a default val cache instance, use `Field(default_factory=...)` rather than `default=`.

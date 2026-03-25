# Task: EfmSnippetView from cache EFM

## Summary
- Added `EfmSnippetView.from_cache_efm(...)` to build a snippet view from offline cache EFM dicts by parsing `__key__` and inferring crop bounds from stored volume metadata.
- Added small bounds parser to accept `points/vol_min|max` and `scene/points/vol_min|max`.
- Added unit test to verify parsing and bounds wiring.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/data_handling/test_dataset.py -k from_cache_efm_parses_key_and_bounds`

## Notes / Follow-ups
- Full `oracle_rri/tests/data_handling/test_dataset.py` run currently fails due to pre-existing issues (missing `load_atek_wds_dataset_as_efm`, `verbose` field validation, missing tar files). Not modified here.

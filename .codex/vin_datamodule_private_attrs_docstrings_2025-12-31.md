# VinDataModule private attribute docstrings

## Summary
- Added class-level docstrings for VinDataModule internal attributes: _train_base, _val_base, _labeler, _train_cache, _val_cache, _train_cache_appender, _val_cache_appender.

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_datamodule.py`

## Notes
- Ruff format failed because `VinDataModuleConfig.train_cache` currently has an incomplete default_factory at line ~308 (syntax error). I did not change that line.

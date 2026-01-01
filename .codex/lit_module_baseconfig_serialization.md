# LitModule BaseConfig Serialization

## Summary
- Added BaseConfig helpers for JSON-serializable dumps and unified Lightning hyperparameter serialization.
- Updated VIN Lightning module summary formatting to use existing `utils.summary.summarize` helper.

## Changes
- `oracle_rri/oracle_rri/utils/base_config.py`: added `model_dump_jsonable` and `to_jsonable` helpers.
- `oracle_rri/oracle_rri/lightning/lit_module.py`: removed local JSON helper, switched to BaseConfig, and refactored `_shape_str` to use `summarize`.
- `oracle_rri/oracle_rri/lightning/lit_datamodule.py`: save hyperparameters using JSONable config dump.

## Notes / Suggestions
- If additional non-JSON types appear in configs (e.g., numpy dtypes), extend `BaseConfig.to_jsonable`.
- Consider reusing `model_dump_jsonable` anywhere else configs are logged or serialized.

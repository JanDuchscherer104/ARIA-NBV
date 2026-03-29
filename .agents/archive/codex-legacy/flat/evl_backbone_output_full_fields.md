# Task: EVL backbone output full-field capture

## Summary
- Expanded `EvlBackboneOutput` to include every output key produced by EVL (voxel features, 2D features/tokens, head outputs, OBB/NMS outputs, and taxonomy metadata) with per-field docstrings.
- Updated `EvlBackbone.forward` to populate all fields from the EVL output dict and attach per-stream 2D features/tokens.
- Extended offline cache serialization to decode `ObbTW`/`TensorWrapper` subclasses correctly.
- Added per-field docstrings to `OracleRriCacheSample`.

## Findings / Potential Issues
- `EvlBackboneConfig.features_mode` is now effectively ignored by `EvlBackbone.forward` (all outputs are captured regardless). If memory pressure becomes an issue, reintroduce a selective capture path that still satisfies “store all fields” when caching is enabled.
- Older cache files will load with new fields defaulting to `None` / `{}`; callers that assume non-`None` values should guard accordingly.

## Suggestions
- Consider a config flag like `capture_all_backbone_fields: bool = True` to make the behavior explicit and to allow selective capture for training-only runs.
- If cache size becomes a problem, add a post-processing hook to drop `voxel_feat` and per-stream 2D features when not needed by downstream models.

## Validation
- `uv run pytest tests/integration/test_vin_real_data.py -q` (from `oracle_rri/`) passed.

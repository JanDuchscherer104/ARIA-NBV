# nbv-summary run fix (2025-12-30)

## Summary
- Fixed scheduler config validation by adding explicit `target` fields to `ReduceLrOnPlateauConfig` and `OneCycleSchedulerConfig`.
- `uv run nbv-summary` now executes successfully and prints the VIN v2 summary + torchsummary output (takes ~2 minutes on this machine).

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_module.py`

## Command run
- `uv run nbv-summary` (from `oracle_rri/`): completed successfully after ~125s.

## Notes
- Summary run is heavy due to oracle labeler + EVL backbone inference; expect ~2 minutes before output on current assets.

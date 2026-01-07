# VIN v2 summarize_vin optional modules

## Summary
- Extended `VinV2ForwardDiagnostics` with optional `pos_grid` and `semidense_feat` fields.
- Wired these fields in `VinModelV2` debug path.
- `summarize_vin` now conditionally reports optional backbone tensors (features, NMS outputs, 2D features) and optional semidense/positional tensors when present.

## Files touched
- `oracle_rri/oracle_rri/vin/types.py`
- `oracle_rri/oracle_rri/vin/model_v2.py`

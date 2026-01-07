# VIN v2 trajectory encoder support

## Summary
- Added optional `traj_encoder` to `VinModelV2Config` and wired `TrajectoryEncoder` into `VinModelV2` feature assembly.
- Added trajectory outputs to `VinV2ForwardDiagnostics` and surfaced them in `summarize_vin` when present.
- Added candidate-to-trajectory cross-attention (MHA) over per-frame trajectory encodings in the reference rig frame; the resulting context is concatenated into VIN v2 features.
- Updated VIN v2 integration test to drop candidate_valid expectations (no longer surfaced in VinPrediction).
- Dropped pooled `traj_feat` from VIN v2 feature concatenation to keep head input dimensions aligned with the trajectory attention context.

## Files touched
- `oracle_rri/oracle_rri/vin/model_v2.py`
- `oracle_rri/oracle_rri/vin/types.py`

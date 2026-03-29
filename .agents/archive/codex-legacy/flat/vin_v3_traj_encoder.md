# Vin v3: optional trajectory encoder

- Added optional trajectory encoder path to `oracle_rri/oracle_rri/vin/model_v3.py` (config flag `use_traj_encoder`, config `traj_encoder`).
- Implemented `_encode_traj_features` (v2 parity) and attention-based `traj_ctx` appended to head input when enabled.
- Extended diagnostics (`VinV3ForwardDiagnostics`) and VIN summary to include trajectory embeddings.
- Added tests for trajectory encoding and forward debug with `use_traj_encoder=True`.

Tests:
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_model_v3_methods.py -k "traj_features or traj_context"`

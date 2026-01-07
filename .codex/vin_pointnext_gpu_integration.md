# VIN v2 PointNeXt integration test (real data)

## Summary
- Ran VIN v2 with PointNeXt-S encoder on a real ASE snippet using GPU.
- Forward pass succeeded when the VIN module was explicitly moved to the GPU.
- Without `vin.to(device)`, the pose LFF stays on CPU and triggers a device mismatch during matmul.

## Command
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python - <<'PY' ...` (manual script using `tests.vin.utils.load_real_snippet_and_cameras`)

## Findings
- The VIN v2 module is initialized on CPU (because backbone is lazily created), so calling `.to(device)` is required before GPU forward.
- Suggestion: restore an internal device sync in `VinModelV2._forward_impl` (e.g., `if next(self.parameters()).device != device: self.to(device)`), or move the model after backbone initialization.
- `PointNeXtSEncoderConfig` is not exported from `oracle_rri.vin` (must import from `oracle_rri.vin.pointnext_encoder`).

## Follow-up
- Implemented internal device sync in `VinModelV2._forward_impl` to auto-move the module to the backbone device.
- Added PointNeXt FiLM-style injection that modulates the global voxel context (with GroupNorm) while keeping the late-fusion concat.

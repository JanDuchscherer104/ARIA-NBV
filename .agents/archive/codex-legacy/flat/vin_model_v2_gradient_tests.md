# VinModelV2 Gradient Tests

## Summary
- Added gradient-flow test for VIN v2 forward/backward with stubbed backbone outputs.

## Tests
- `pytest oracle_rri/tests/vin/test_vin_model_v2_gradients.py`

## Results
- Passed. Warnings from torchmetrics + torch jit deprecations surfaced.

## Notes / Suggestions
- Consider adding a test that masks invalid candidates and checks `candidate_valid` gating logic.
- Add a numerical check that `pose_log_scale` gradients respond to scale changes (e.g., perturb and assert loss diff).

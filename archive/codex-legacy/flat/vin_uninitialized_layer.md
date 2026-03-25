# VIN uninitialized layer warning

## Findings
- The PyTorch Lightning warning about `UninitializedParameter` came from lazy layers in `oracle_rri/oracle_rri/vin/model.py`:
  - `nn.LazyLinear` in `VinScorerHead` when `in_dim=None`.
  - `nn.LazyConv3d` in `VinModel.field_proj`.
- These were only uninitialized until the first forward pass; the model summary runs before that.

## Changes
- Replaced `LazyConv3d` with `Conv3d` using `in_channels=len(scene_field_channels)`.
- Passed explicit `head_in_dim = pose_encoder_sh.out_dim + field_dim (+ field_dim if global pool)` when building `VinScorerHead`.

## Tests
- `pytest tests/vin/test_vin_model_integration.py -m integration` (using `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python`) failed:
  - `TypeError: VinModel.forward() takes 2 positional arguments but 3 positional arguments (and 2 keyword-only arguments) were given`.
  - The test passes `t_w_c` positionally; `VinModel.forward` requires `candidate_poses_world_cam` as keyword-only.

## Suggestions
- Update the test to call `vin(sample.efm, candidate_poses_world_cam=t_w_c, ...)` or change `VinModel.forward` to accept a positional argument if you want the test to remain as-is.

# Remove `_VinSummaryWrapper` from VIN torchsummary (2025-12-24)

## Context
`VinLightningModule.summarize_batch()` optionally appends `torchsummary` output for:
- the full VIN model (`self.vin`)
- selected submodules (`pose_encoder_sh`, `field_proj`, `head`)

The full VIN summary previously used an internal `_VinSummaryWrapper` that:
- accepted a dummy tensor to satisfy `torchsummary.summary(input_data=...)`
- called `VinModel.forward(...)` with the real EFM batch/candidate inputs stored on the wrapper

## Problem
Running `torchsummary` with the wrapper triggers a real VIN forward pass during summary generation.
This is redundant (the function already prints tensor shapes from a prior `forward_with_debug`) and can fail
(e.g. the reported `KeyError: 0` originating from the wrapper-driven forward path).

## Change
- Removed `_VinSummaryWrapper` entirely.
- Switched the VIN-level `torchsummary` call to `input_data=None` so torchsummary only traverses modules and
  reports parameter counts without executing `VinModel.forward`.

The per-submodule torchsummary calls remain unchanged (they run on already-materialized debug tensors).

## Tests
- `python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py -q`

## Follow-ups
- If we ever need VIN I/O sizes in the torchsummary table, prefer adding a lightweight wrapper that exercises
  only the trainable path (pose enc + head) rather than full EVL/voxel querying, or switch to a summary tool
  that supports non-tensor/keyword-only inputs.

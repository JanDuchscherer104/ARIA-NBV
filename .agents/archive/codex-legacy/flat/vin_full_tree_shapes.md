# VIN full module tree with shapes (2025-12-24)

## Goal
Show the full VIN `nn.Module` tree **with output shapes** on **real data** inside `VinLightningModule.summarize_batch()`.

## Why `torchsummary(self.vin, input_data=None)` had no shapes
`torchsummary` only reports input/output shapes if it executes a forward pass.
With `input_data=None`, it skips the forward pass and can only print parameter counts.

## Why we did not reintroduce `_VinSummaryWrapper`
`VinModel.forward(...)` takes non-tensor inputs (`efm` dict + PoseTW + PyTorch3D cameras), so `torchsummary` can’t call it directly.
A wrapper can “smuggle” these inputs via attributes/dummy tensors, but that forces an extra full VIN forward (and previously hit a `KeyError: 0` in some runs).

## Implementation
- Added `_ModuleOutputShapeRecorder` + `_render_module_tree_with_shapes` in `oracle_rri/oracle_rri/lightning/lit_module.py`.
- During the existing `forward_with_debug(...)` call in `summarize_batch()`, we attach forward hooks to all VIN submodules and record their **observed output shapes**.
- The “VIN module tree (real-data shapes)” section renders the module hierarchy up to `torchsummary_depth`, with:
  - `out=[...]` (batch dim normalized to `-1`)
  - `params=...` (direct, non-recursive params; `--` when zero)

This avoids a second VIN forward and avoids wrapper-based input hacks.

## Test
- `python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py -q`

## Notes
- Shapes are “observed” for the specific batch; modules not executed would show `out=?`.

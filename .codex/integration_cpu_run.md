# Integration runs on CPU (num_workers=1 request)

## What I tried
- Ran integration tests on CPU with `CUDA_VISIBLE_DEVICES=""` to force CPU.
  - `oracle_rri/tests/integration/test_vin_real_data.py` failed: VIN v1 requires `backbone_out` when the backbone is disabled.
  - `oracle_rri/tests/integration/test_vin_v2_real_data.py` failed: EVL backbone uses xFormers `memory_efficient_attention`, which is CUDA-only.

## Errors observed
- VIN v1: `RuntimeError: backbone_out is required when the VIN backbone is disabled.`
- VIN v2: `NotImplementedError: No operator found for memory_efficient_attention_forward ... device=cpu` (xformers only supports CUDA for this op).

## Constraints for num_workers=1 on CPU
- Online oracle labeler requires `num_workers=0` (already enforced in `VinDataModuleConfig`).
- CPU forward passes require either:
  1) a GPU to compute EVL backbone outputs, or
  2) a precomputed offline cache with `backbone_out` available.
- There is no `offline_cache/` present at the default path, so CPU-only forward passes with cached backbone outputs aren’t currently possible.

## Next steps suggested
- Run the integration tests on GPU (xformers ops are CUDA-only).
- Or provide an offline cache that includes backbone outputs so we can run CPU-only forward passes with `num_workers=1`.

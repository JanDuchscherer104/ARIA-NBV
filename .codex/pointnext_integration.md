# PointNeXt-S optional semidense encoder

## Summary
- Added optional PointNeXt-S adapter in `oracle_rri/oracle_rri/vin/pointnext_encoder.py` using the vendored `external/PointNeXt` OpenPoints stack.
- Subsamples semidense points to `max_points` (default 3000) via `EfmPointsView.collapse_points` before encoding.
- Added optional config hook (`VinModelV2Config.point_encoder`) and integrated semidense embedding into the VIN v2 head inputs.
- Switched semidense extraction to `EfmSnippetView.from_cache_efm(efm).semidense.collapse_points()` to match cache data paths.
- Updated `docs/contents/impl/vin_v2_feature_proposals.qmd` and `docs/references.bib` with PointNeXt/OpenPoints references.

## Notes / Suggestions
- If using the PointNeXt-S encoder, ensure `external/PointNeXt` is available and provide the model zoo YAML + checkpoint path; otherwise the adapter raises a clear error.
- PointNeXt layers rely on CUDA-only ops (pointnet2/pointops). The adapter now raises if you try to run on CPU; run on GPU with compiled ops.
- Consider adding a config flag for candidate-conditioned point encoding (camera-frame points) if global embeddings prove too coarse.
- Running the integration tests requires the real ASE snippet fixture; ensure the test assets are available before relying on CI.
- Official OpenPoints model zoo provides PointNeXt-S checkpoints via Google Drive folders (e.g., S3DIS, ScanObjectNN, ModelNet40); wire `checkpoint_url` + auto-download to handle Drive folders (gdown-style) when adding default weights.

## Integration Tests / Training
- Integration test (real data) succeeded with `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_v2_integration.py`.
- Full epoch run with default dataloader workers failed due to shared memory (bus error, no space on device).
- Successful 1-epoch run required `--datamodule-config.num-workers 0` and `--trainer-config.limit-train-batches 5` to complete within time.

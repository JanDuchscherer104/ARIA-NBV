# Offline cache backbone pruning (2026-01-03)

## Summary
- Added `backbone_keep_fields` to `OracleRriCacheDatasetConfig` and wired it through `decode_backbone` to skip decoding unused EVL fields.
- `VinDataModuleConfig` now exposes a default backbone field allowlist for VinModelV2 training, and propagates it to train/val cache configs.
- `decode_dataclass` supports an `include_fields` filter to avoid decoding unused dataclass fields (defaults still apply).

## Potential issues
- `torch.load` still deserializes the entire payload; pruning avoids device transfers/retention but not disk IO.
- If `backbone_keep_fields` omits required fields (e.g., `t_world_voxel`, `voxel_extent`), decoding will raise.

## Suggestions
- Split cached backbone outputs into separate files to truly avoid loading unused tensors.
- Add a small validation helper to ensure `backbone_keep_fields` covers `VinModelV2` requirements derived from config.

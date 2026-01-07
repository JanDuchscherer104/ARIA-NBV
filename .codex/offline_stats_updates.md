Task: Extend offline stats to include VIN batch shapes + cache discrepancy checks.

Changes:
- Added `VinOracleBatch.shape_summary()` for tensor shape reporting.
- Offline cache stats now limit backbone fields to avoid OBB batching errors.
- Offline stats page shows VIN batch shapes (raw vs padded), vin snippet shapes, and cache discrepancy tables.

Notes:
- Oracle vs VIN cache discrepancies are computed using snippet tokens to normalize IDs.
- Offline stats now broadcasts reference pose tensors to match batched candidate dims.
- Broadcast fix pads singleton dims at the tail (e.g., (B,) -> (B,1)) to match (B,N).

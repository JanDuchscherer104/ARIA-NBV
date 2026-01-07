# Attribution: huber-based candidate selection + multi-sample aggregation

## Summary
- Added candidate selection modes based on per-candidate Huber error (min/max) between predicted and oracle RRI.
- Added multi-sample attribution aggregation for offline cache (contiguous sample range).
- Added optional attribution std display and candidate index list for aggregated runs.

## Files changed
- `oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Notes
- Multi-sample aggregation is enabled only for offline cache sources and uses contiguous indices starting at `Cache index`.
- Huber uses PyTorch SmoothL1 with default beta=1.0, matching training.

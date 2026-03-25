# Remove VIN v2 validity weighting

## Changes
- Removed VIN v2 center-based validity sampling (`_compute_candidate_validity`) and related dataclass.
- VIN v2 now returns `candidate_valid` and `valid_frac` as all-true / all-ones placeholders to preserve API shape without influencing training.
- VIN v2 summary output no longer reports a validity section.
- W&B VIN aux metrics list no longer expects `valid_frac_mean` or `candidate_valid_fraction`.
- Restored/kept `VinPrediction` validity fields to match existing v1/v2 call sites.

## Rationale / findings
- The previous v2 “validity” was a single-voxel `counts_norm` lookup at the candidate center, which was not a frustum coverage proxy and was unused for features.
- Loss weighting by that proxy was already removed in the current Lightning module, so the v2 sampling chain was pure overhead.

## Suggestions / future work
- If a validity/coverage proxy is needed again, compute it from frustum token validity (v1-style) or from projected semidense points, not from the candidate center voxel alone.
- Consider making UI validity stats conditional on model capability (e.g., hide when using v2 placeholders).

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_v2_integration.py -m integration`
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_integration.py -m integration` (failed: requires backbone_out when VIN backbone disabled).

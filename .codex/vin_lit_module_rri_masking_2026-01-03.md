# VIN Lightning RRI masking (2026-01-03)

- Added finite-mask gating in `VinLightningModule._step` to skip non-finite RRI entries and prevent NaN loss/logs.
- Mask now applied to logits/labels/probs/aux loss/metrics; batches with no valid RRI are skipped (logged as `*/skip_no_valid`).
- Added unit tests covering masking and full-NaN skip in `tests/lightning/test_lit_module_masking.py`.
- Real-data integration test run: `tests/vin/test_vin_model_v2_integration.py -m integration` (PASS).
- Known unrelated failure: `oracle_rri/tests/integration/test_vin_lightning_real_data.py` uses removed `candidate_chunk_size` in `OracleRRIConfig`.

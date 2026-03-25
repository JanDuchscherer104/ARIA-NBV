# VIN metrics additions (coral loss baseline + top-3 accuracy)

## Summary of changes
- Added `coral_random_loss()` utility and exported it for reuse.
- Added `topk_accuracy_from_probs()` helper and exposed it in `rri_metrics`.
- Logged `coral_loss_rel_random` (ratio vs random CORAL baseline) and `top3_accuracy` in the VIN Lightning module.
- Documented the new metrics in `docs/contents/impl/vin_nbv.qmd`.
- Added pytest coverage for the new helper functions.

## Potential issues / caveats
- `top3_accuracy` is computed per-step from candidate-level probabilities and logged via Lightning aggregation; if you want a strict epoch-level metric (global top-k accuracy), consider moving it into `VinMetrics` as a stateful torchmetric.
- `coral_random_loss()` clamps `num_classes` to at least 1 (matching the Streamlit panel). If `num_classes=1` is ever used, the baseline is only a guard value and not a meaningful CORAL reference.
- The integration test `oracle_rri/tests/integration/test_vin_lightning_real_data.py` currently fails early because `RriOrdinalBinner` is not exported from `oracle_rri.vin` (ImportError). This is unrelated to the metric changes but blocks real-data smoke testing.

## Suggestions / future improvements
- Add additional ranking metrics (top-k recall, Kendall/Spearman per candidate set) as stateful metrics for more stable epoch-level reporting.
- Log the random baseline itself (e.g. `coral_loss_random`) alongside the ratio for easier dashboard comparison.

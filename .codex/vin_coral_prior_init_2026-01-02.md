## CORAL prior-logit init + OneCycle recipe (2026-01-02)

### What changed
- Added `bin_counts` to `RriOrdinalBinner` and serialize it in JSON for empirical priors.
- Added `class_priors()` / `threshold_priors()` helpers on the binner.
- Added `CoralLayer.init_bias_from_priors(...)` to initialize CORAL biases from cumulative priors.
- Added `VinLightningModuleConfig.coral_bias_init` (default `default`, new `prior_logits` option) and a setup hook to apply it.
- Updated `.configs/offline_only.toml` to Recipe‑2 settings:
  - `num_classes=15` (top‑level and VIN head),
  - optimizer LR `3e‑4`,
  - OneCycle scheduler (`max_lr=1e‑3`, `pct_start=0.15`),
  - aux decay `gamma=0.90`,
  - `coral_bias_init="prior_logits"`.

### Notes / follow‑ups
- Existing `rri_binner.json` files won’t have `bin_counts`; prior init falls back to uniform priors.
  Re‑fit the binner to populate `bin_counts` if you want true empirical priors.

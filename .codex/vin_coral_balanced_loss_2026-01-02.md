## CORAL balanced threshold loss (2026-01-02)

### What changed
- Added configurable CORAL loss variants: `coral`, `balanced_bce`, `focal`.
- Implemented per-threshold balancing via `pos_weight` derived from priors (binner or batch).
- Implemented focal loss option with per-threshold alpha (default inferred from priors).
- Added config knobs in `VinLightningModuleConfig` and wired the new loss path.
- Set `.configs/offline_only.toml` to use `balanced_bce` with `binner` priors.

### Notes
- Torchmetrics doesn’t provide a built-in CORAL loss; the implementation uses `binary_cross_entropy_with_logits`.
- For true empirical priors in `balanced_bce`, re-fit the binner to populate `bin_counts` in `rri_binner.json`.

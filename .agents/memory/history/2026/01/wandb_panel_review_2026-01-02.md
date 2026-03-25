---
id: 2026-01-02_wandb_panel_review_2026-01-02
date: 2026-01-02
title: "Wandb Panel Review 2026 01 02"
status: legacy-imported
topics: [wandb, panel, 2026, 01, 02]
source_legacy_path: ".codex/wandb_panel_review_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# W&B panel review for VIN v2 (2026-01-02)

## Findings
- W&B panel calibration and train/val pairing logic only sees `train/` + `val/` prefixes, but VIN logs RRI metrics under `train-aux/` + `val-aux/`; calibration plots cannot resolve `pred_rri_mean` or `rri_mean` as currently written.
- Confusion matrix/label histogram media keys are logged under `train-figures/*` and `val-figures/*` (from `VinLightningModule._log_figure`), while the panel searches `train/*` and `val/*` only.
- Attribution explorer is currently disabled in the panel, and the generic `interpretability/attribution.py` assumes 2D conv/image inputs; it needs VIN-aware adapters (feature attribution, 3D field attribution) to analyze `VinModelV2` outputs.

## Proposed revisions (high level)
- Extend metric pairing helpers to recognize `train-aux`/`val-aux` and update calibration/gap sections accordingly.
- Update media key search to include `*-figures/confusion_matrix(_step)` and `*-figures/label_histogram(_step)` in addition to existing keys.
- Add a VIN attribution section that:
  - Loads a checkpoint from `PathConfig.checkpoints`.
  - Builds a `VinLightningModule` from checkpoint hyperparameters.
  - Pulls a real batch from offline cache.
  - Uses `VinModelV2.forward_with_debug` and an adapted `AttributionEngine` to compute feature attributions for candidate scores.

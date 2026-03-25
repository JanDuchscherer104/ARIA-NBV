---
id: 2026-01-04_wandb_export_2026-01-04_inspection
date: 2026-01-04
title: "Wandb Export 2026 01 04 Inspection"
status: legacy-imported
topics: [wandb, export, 2026, 01, 04]
source_legacy_path: ".codex/wandb_export_2026-01-04_inspection.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# W&B export inspection (2026-01-04)

Source: `.codex/wandb_export_2026-01-04T19_22_34.107+01_00.csv` (project-level run export with configs).

## Key finding: OneCycleLR misconfig in latest runs

Runs with OneCycle-like params show:

| run name | max_lr | div_factor | final_div_factor | implied init_lr | implied final_lr |
|---|---:|---:|---:|---:|---:|
| JanPC | 1e-3 | 25 | **1e-4** | 4e-5 | **0.4** |
| fix-masks | 1e-3 | 25 | **1e-4** | 4e-5 | **0.4** |
| coral-fixed-masks | 1e-3 | 25 | 1e4 | 4e-5 | 4e-9 |
| bce+frustum-sampling | 1e-3 | 25 | 1e4 | 4e-5 | 4e-9 |
| one-cycle-higher-grad-clip | 1e-3 | 25 | 1e4 | 4e-5 | 4e-9 |

`final_div_factor = 1e-4` causes the LR to *explode* at the end of the cycle (final_lr ~ 0.4 for max_lr=1e-3), which matches the LR plot rising toward 1e-2+.

## Actionable fix

- Ensure `final_div_factor >= 1`, preferably `1e4`, for OneCycleLR.
- Re-run `JanPC` / `fix-masks` with the corrected config.

## Optional tuning for faster early dynamics

If the early slope still looks slow (only ~4 epochs into a 50-epoch run):

- Decrease `pct_start` (e.g., `0.08`) to reach `max_lr` earlier.
- Reduce `div_factor` (e.g., `10`) to raise the initial LR.
- Keep `final_div_factor = 1e4` to avoid end-of-cycle blow-up.

## Other likely contributors to poor training dynamics (beyond LR blow-up)

1) **Warmup too long for 50 epochs**  
   With `pct_start=0.15`, the LR keeps rising for ~7.5 epochs. If you only look at the first ~4 epochs, the loss slope will appear flatter vs plateau runs.

2) **Gradient clipping loosened (3 → 8)**  
   Raising `gradient_clip_val` can allow spikes to pass through. This often shows up as sudden loss jumps even before the LR peak.

3) **Metric comparability across runs**  
   `train/coral_loss_rel_random_*` depends on `num_classes` (random baseline scales with `K-1`). If runs used `K=8/30`, slopes are not directly comparable.

4) **Model complexity drift**  
   The “simpler” runs used fewer `scene_field_channels` and smaller heads. Adding `free_input/new_surface_prior`, PointNeXt features, or larger heads can add noise if the signals are not well calibrated.

5) **Binner mismatch / stale thresholds**  
   Reusing a `rri_binner.json` fit on different data (or a different `num_classes`) can produce poor ordinal targets and unstable loss.

6) **Cache distribution drift**  
   If the offline cache was rebuilt after candidate-generation changes (e.g., CW90 correction), the label distribution may have shifted. Mixing older caches with newer configs increases noise.

7) **OneCycle schedule sensitive to total steps**  
   Changes in dataset length or filtering alter the inferred `total_steps`, which reshapes the LR curve even with identical hyperparameters.

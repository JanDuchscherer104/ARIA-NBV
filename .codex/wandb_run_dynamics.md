# W&B run dynamics (top-2) — paper + slides sync

Date: 2026-01-30

## Goal

Update the paper + slides to summarize the **within-run** training improvement for the current top-2 W&B runs and include **first vs final validation confusion matrices** for both runs. Keep the narrative focused on dynamics (not config-diff attribution).

## Runs (W&B: `traenslenzor/aria-nbv`)

- **`hq1how1j`** — run name: `R2026-01-27_13-08-02`
  - Trajectory encoder: disabled
  - Scheduler: ReduceLROnPlateau
- **`rtjvfyyp`** — run name: `v03-best`
  - Trajectory encoder: enabled
  - Scheduler: OneCycleLR

## Start → finish improvements (first & last logged points)

Computed via W&B API `run.scan_history(keys=[...])` (filtering out NaNs) and written to:

- `docs/typst/slides/data/wandb_top2_improvements.json`

Metrics included:

- `train/coral_loss_rel_random_step`
- `val/coral_loss_rel_random`
- `val-aux/spearman`
- `val-aux/top3_accuracy`

## Confusion matrices (first & final)

Source (per run):

- `.logs/wandb/wandb/run-*/files/media/images/val-figures/`

Copied to (paper/slides assets):

- `docs/figures/wandb/<run_id>/val-figures/`

Stable aliases added for Typst:

- `docs/figures/wandb/hq1how1j/val-figures/confusion_start.png`
- `docs/figures/wandb/hq1how1j/val-figures/confusion_end.png`
- `docs/figures/wandb/rtjvfyyp/val-figures/confusion_start.png`
- `docs/figures/wandb/rtjvfyyp/val-figures/confusion_end.png`

## Paper edits

- `docs/typst/paper/sections/09c-wandb.typ`
  - Added a start→finish table for both runs.
  - Added confusion-matrix figures (start vs finish) for both runs.
  - Clarified that shared curve plots are shown for `rtjvfyyp` and that trajectory attribution is **inconclusive**.
- `docs/typst/paper/sections/07b-training-config.typ`
  - Explicitly tagged the baseline training-config table as `hq1how1j` / `R2026-01-27_13-08-02`.
- `docs/typst/paper/sections/11-conclusion.typ`
  - Replaced stale hard-coded metrics with artifact-driven values:
    - offline-cache stats from `docs/typst/slides/data/offline_cache_stats.json`
    - best-run metrics from `docs/typst/slides/data/wandb_rtjvfyyp_summary.json`

## Slide edits

- `docs/typst/slides/slides_4.typ`
  - Added `wandb_top2` import from `docs/typst/slides/data/wandb_top2_improvements.json`.
  - Added two slides:
    - “Top-2 runs: start → finish improvements”
    - “Val confusion matrices: collapse → structure”
  - Softened trajectory claims: no strong attribution without a controlled ablation.

## Regeneration snippets

Recreate `wandb_top2_improvements.json` from API:

```bash
oracle_rri/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import wandb

ENTITY_PROJECT = "traenslenzor/aria-nbv"
RUN_IDS = ["hq1how1j", "rtjvfyyp"]
KEYS = [
    "train/coral_loss_rel_random_step",
    "val/coral_loss_rel_random",
    "val-aux/spearman",
    "val-aux/top3_accuracy",
]

api = wandb.Api()

def first_last(run, key):
    vals = []
    for row in run.scan_history(keys=[key]):
        v = row.get(key)
        if v is None:
            continue
        if v != v:
            continue
        vals.append(float(v))
    return vals[0], vals[-1]

out = {}
for run_id in RUN_IDS:
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    metrics = {}
    for key in KEYS:
        s, e = first_last(run, key)
        metrics[key] = {"start": s, "end": e, "delta": e - s, "delta_pct": 100 * (e - s) / s}
    out[run_id] = {"name": run.name, "run_id": run.id, "url": run.url, "metrics": metrics}

Path("docs/typst/slides/data/wandb_top2_improvements.json").write_text(
    json.dumps({"entity_project": ENTITY_PROJECT, "keys": KEYS, "wandb_top2": out}, indent=2, sort_keys=True) + "\n"
)
PY
```

## Open questions / follow-ups

- Run a **controlled** traj ablation: toggle only `use_traj_encoder`, keep schedule/seed/data fixed, and compare across multiple seeds.
- Consider wiring the paper’s W&B table to `wandb_top2_improvements.json` (artifact-driven numbers) to prevent future drift.

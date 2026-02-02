# Paper: Appendix-Extra Metrics (2026-01-30)

## Goal
Resolve the TODOs in `docs/typst/paper/sections/12b-appendix-extra.typ` and fix the paper render issues by:

- Match the *actual* Lightning/W&B logging keys used by VINv3.
- Reuse the shared symbol/equation dictionary in `docs/typst/shared/macros.typ`.
- Avoid layout breakage from large tables / inline formulas.

## What I changed

### 1) Logging namespaces & key patterns (ground truth)
Verified by reading:

- `oracle_rri/oracle_rri/rri_metrics/logging.py` (enums + key composition)
- `oracle_rri/oracle_rri/lightning/lit_module.py` (manual robustness keys, figures, grad norms)

Key patterns used in docs now reflect:

- `stage/<loss>` (main loss scalars), `stage ∈ {train,val,test}`
- `stage-aux/<metric>` (aux/diagnostic scalars)
- `stage-figures/<tag>` (confusion matrices + label histograms logged as images)
- `train-gradnorms/grad_norm_<module>` (grad norms; train only)
- robustness flags in `stage/...`: `drop_nonfinite_logits_frac`, `skip_nonfinite_logits`, `skip_no_valid`

Also clarified Lightning’s `_step` / `_epoch` suffix behavior and that some metrics are explicitly step-only (e.g. `train-aux/spearman_step`).

### 2) `12b-appendix-extra.typ` layout
The rendered tables were too large and broke across columns/pages (and formulas in table cells made it worse), so I:

- Replaced the two big tables with a concise grouped bullet list of keys.
- Added `breakable-key(...)` + `log-key(...)` so long keys can line-break cleanly.
- Added a separate “Definitions (selected)” subsection with equation blocks pulled from `#eqs.*` (and labeled so the bullets can say “Eq. (..)” without embedding formulas inline).
- Kept the missing diagnostics covered (confusion matrix / label histogram figures, robustness skip flags, focal loss).
- Kept the grad-norm implementation note (target selection via `group_depth` / `include` / `exclude` / `max_items` and norm type).

### 3) Internal DB update
Added a concise bullet block to `.codex/AGENTS_INTERNAL_DB.md` summarizing the VIN logging namespaces/keys so future work doesn’t drift.

## Validation
- `make typst-paper` succeeds after the edits.

## Follow-ups (not done here)
- Several other paper sections still contain `TODO(paper-cleanup)` notes about reusing `#eqs.metrics.*` instead of retyping (e.g. `docs/typst/paper/sections/09a-evaluation.typ`). Consider addressing those in a dedicated “paper cleanup” pass to keep scope tight.
- `semidense_valid_frac_*` is currently logged as an alias of `semidense_candidate_vis_frac_*` (both map to the same tensor in `VinLightningModule`). If this is accidental, simplify the code or document it in the logging module.

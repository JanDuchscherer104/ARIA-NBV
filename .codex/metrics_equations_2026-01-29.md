# VIN Lightning metrics equations (2026-01-29)

## Scope
- Reviewed metrics and losses logged in `oracle_rri/oracle_rri/lightning/lit_module.py` plus `rri_metrics/logging.py`.
- Added missing notation + equations to `docs/typst/shared/macros.typ`.
- Added appendix equations + missing metric rows in `docs/typst/paper/sections/12b-appendix-extra.typ`.
- Added slides appendix formulas in `docs/typst/slides/slides_4.typ`.

## Key findings
- Logged diagnostics in `lit_module.py` include non-enum keys: `drop_nonfinite_logits_frac`, `skip_nonfinite_logits`, `skip_no_valid`, and `train-gradnorms/grad_norm_*`. These were missing from the appendix tables.
- Several logged metrics lacked explicit equation definitions: bias/variance, coverage weights/strength schedules, balanced/focal CORAL variants, aux-regression decay, and validity/skip diagnostics.

## Changes applied
- **Macros**: added symbols for predicted RRI, coverage/weights, validity masks; added equations for balanced/focal CORAL, aux regression (MSE/Huber + Huber definition), aux-weight decay, coverage strength schedules, bias/variance, validity means/std, drop/skip metrics, label histogram.
- **Paper**: expanded metrics table with the non-enum log keys; added a two-column equation block covering losses + diagnostics (appendix).
- **Slides**: added two appendix slides (“Loss + weighting equations”, “Metric equations”) covering all logged metrics.

## Verification
- `typst compile docs/typst/slides/slides_4.typ /tmp/slides_4.pdf --root docs`
- `typst compile docs/typst/paper/main.typ /tmp/paper.pdf --root docs`

## Suggestions / follow-ups
- If we want to avoid dense appendix slides, consider moving the full equation blocks into a dedicated “Metrics appendix” deck and link from the main slides.
- Consider aligning notation for threshold priors vs. class marginals (both use `pi_k` in different contexts) if it becomes confusing in the paper.

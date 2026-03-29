## Task: Training — Objective, Metrics & Diagnostics (slides + paper)

Date: 2026-01-30

### Goal
- Update the presentation section **“Training: Objective, Metrics & Diagnostics”** to:
  1) introduce metric definitions alongside the corresponding plots,
  2) show the within-run start→finish summary for the best run,
  3) then present the (uncontrolled) comparison to the baseline/ablation run.
- Ensure **paper section `09c-wandb.typ` contains only `rtjvfyyp`**, and move the
  cross-run comparison into `09b-ablation.typ`.
- Use **symbol macros** from `docs/typst/shared/macros.typ` instead of long W&B key names.

### Data / runs referenced
- Best run: `rtjvfyyp` (`v03-best`)
- Baseline (comparison): `hq1how1j` (`R2026-01-27_13-08-02`)

### Key outcome: which run is better?
`rtjvfyyp` performs better on the final validation metrics:
- Lower relative CORAL loss: `0.666` vs `0.677`
- Higher Spearman: `0.501` vs `0.469`
- Higher Top-3: `0.329` vs `0.314`

### Paper changes (Typst)
- `docs/typst/paper/sections/09c-wandb.typ`
  - Now focuses on **within-run** improvements for `rtjvfyyp` only.
  - Uses macro-based symbols (e.g., `#(symb.vin.loss)_("rel")`, `rho`, `TopKAcc(3)`).
  - Includes training curves and `rtjvfyyp` start/finish confusion matrices.
- `docs/typst/paper/sections/09b-ablation.typ`
  - Holds the **uncontrolled** comparison (explicitly labeled as inconclusive).
  - Uses symbols in the “top-2 final metrics” table header.
  - Includes `hq1how1j` confusion matrices and references the `rtjvfyyp` figure.

### Slide changes (Typst)
File: `docs/typst/slides/slides_4.typ` (training section only; earlier slides remain unchanged)

**New/updated ordering**
1) Training objective (minimal; avoids repeating full CORAL derivations)
2) Best-run curves + metrics (definitions placed next to the plots)
3) Auxiliary regression (loss + schedule; symbol-based text)
4) Best run start→finish summary table
5) Best vs baseline comparison (table + start/finish confusion matrices; labeled “not controlled”)

**Layout fixes**
- Removed the previous “Metrics & diagnostics” slide which overflowed onto two pages.
- Fixed a Typst formatting bug where `strong[...]` was being rendered literally in the comparison slide.

### Figures added/used in slides
- Curves: `docs/figures/wandb/*.png` (same as paper training-dynamics figures).
- Confusion matrices: `docs/figures/wandb/{hq1how1j,rtjvfyyp}/val-figures/confusion_{start,end}.png`.

### Notes / follow-ups
- If we want to avoid slide figure numbering (“Figure 35: …”), switch slide figures to caption-less `#image(...)` or use a custom caption style for slides.
- If we add Spearman vs step curves to W&B exports, they can be included next to the Top-3 plot for a more complete “ranking” story.


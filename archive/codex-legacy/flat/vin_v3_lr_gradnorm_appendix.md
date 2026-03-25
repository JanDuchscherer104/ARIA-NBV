# VINv3 LR + Gradnorm Appendix Update (2026-01-26)

## Scope
Added LR/gradnorm diagnostics to the VINv3 streamline appendix and removed the top-trials table.

## W&B Findings (vin-v3-01 vs T41)
- **LR schedule**: v3 peaks later and lower.
  - vin-v3-01: lr-AdamW first 2.0e-5, max 6.61e-5 at step 353 (no decay in logged window).
  - T41: lr-AdamW first 4.12e-6, max 1.83e-4 at step 32, decays to 1.3e-8 by step 266.
- **Grad norms (early mean)**: v3 is 6–16x smaller in core modules.
  - head_mlp: 0.63 (v3) vs 6.86 (T41)
  - head_coral: 0.83 vs 5.64
  - global_pooler: 0.06 vs 0.68
  - field_proj: 0.01 vs 0.22
  - pose_encoder_lff: 0.02 vs 0.09
  - sem_proj_film: 0.015 vs 0.097
- **Extra gradient paths in T41**: trajectory, frustum, and point encoder norms are large early (e.g., traj_attn ~3.08, point_encoder ~8.09), providing candidate-specific gradient energy that v3 lacks.

## Interpretation
Lower and delayed LR peak + much smaller grad norms in v3 likely reduce early learning pressure and exacerbate mode-collapse. The objective appears less responsive without candidate-specific modules.

## Files Updated
- `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ`

## Build Check
- `typst compile --root docs docs/typst/paper/main.typ`

## Follow-ups
- If needed, add a lightweight table or CSV-based appendix for LR/gradnorm numeric summaries.

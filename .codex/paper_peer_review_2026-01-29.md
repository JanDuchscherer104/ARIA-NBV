# Paper peer review notes (2026-01-29)

## High-level
- Overall structure matches a publishable ML paper (problem → dataset → method → objective → evaluation → diagnostics).
- Main gaps: stronger quantitative results in main sections, clearer separation of *implemented* vs *planned* components, and tighter linkage between diagnostics and actionable decisions.

## Section-by-section highlights (condensed)
- 01 Introduction: Strong motivation; add 1–2 concrete contributions + explicit limitations.
- 02 Related Work: Solid coverage; consider brief comparison table or sharper contrast with VIN-NBV/GenNBV.
- 03 Problem Formulation: Clean; explicitly define candidate set and constraints early.
- 04 Dataset: Good; add exact counts used for current experiments + mesh subset stats.
- 05 Coordinate Conventions: Clear; add one schematic figure for frames (if available).
- 05 Oracle RRI: Good pipeline description; add complexity/cost estimates per snippet.
- 06 Architecture: Detailed; add a short “what is actually trained” paragraph and parameter count summary.
- 07 Training Objective: Solid; add concrete hyperparameters for CORAL bins and aux loss.
- 07a Binning: Good; add rationale for stage normalization if used in experiments.
- 07b Training Config: Mark as future work is good; but include current config elsewhere.
- 08 System Pipeline: Clear; consider a single end‑to‑end figure.
- 08a Frustum Pooling: Good but v3 uses projection stats; clarify current status vs v2.
- 09 Diagnostics: Valuable; add examples linking a failure mode to a fix.
- 09a Evaluation: Good; include primary metrics used in current runs.
- 09b Ablation: Currently mostly plan; add any completed ablation or explicitly mark as pending.
- 09c W&B: Good; include 1–2 plots in main text (now in appendix).
- 10 Discussion: Strong; add concise “what we would do next” bullets.
- 10a Entity-aware: Good vision; label clearly as future work.
- 11 Conclusion: Solid; add 1–2 quantitative outcomes even if preliminary.
- Appendices: Comprehensive; new slide-figure gallery added.

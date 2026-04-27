---
id: 2026-03-31_thesis_outlook_modality_symbol_alignment
date: 2026-03-31
title: "Thesis Outlook Modality Symbol Alignment"
status: done
topics: [slides, typst, notation, rl]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/shared/macros.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/thesis_symbol_fix/page.png
assumptions:
  - The theory slide should use concrete modality symbols for history tuples while keeping `#symb.rl.hist_ego` / `#symb.rl.hist_cf` as bundle labels.
---

Task:
- Replace generic `O^(...)` placeholders in the theory-slide observation-space definitions with concrete symbols for images, poses, point clouds, and geometry-only counterfactual modalities.

Method:
- Added a new `symb.obs` namespace in `docs/typst/shared/macros.typ` for RGB/grayscale images, depth, pose stream, pose metadata, semidense and counterfactual point clouds, generic visibility, look-at target, voxel-grid observations, face visibility, voxel centers, and face normals.
- Rewrote `#eqs.rl.hist_ego` and `#eqs.rl.hist_cf` to use those concrete symbols plus the existing EVL field symbol `#(symb.vin.field_v)`.
- Recompiled both `slides_thesis_outlook.typ` and `slides_advisor_delta.typ` to confirm the shared-macro change stayed compatible.

Findings:
- The theory slide now defines the history bundles as:
  - ego: `(I^rgb, X, P^semi, F_v^ego)_(1:t)`
  - counterfactual: `(O^ego_(1:t), (D^cf, V^cf, P^cf)_(1:t))`
- The updated math remains legible at the current slide scale without overflow.

Verification:
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_advisor_delta.typ /tmp/slides_advisor_delta_check.pdf --root .`
- visual inspection of `/tmp/thesis_symbol_fix/page.png`

Canonical state impact:
- None. This is notation cleanup for shared docs/slides.

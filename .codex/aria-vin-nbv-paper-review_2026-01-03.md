# Aria-VIN-NBV Paper Review (2026-01-03)

## Build status
- Typst compile: `typst 0.14.1`, `--root docs`, output `docs/typst/paper/main.pdf`.
- Page count: 16 pages.
- Visual inspection: rendered pages `1–16` to PNG and reviewed for ordering, layout, and legibility.

## Compressed architecture memory (for future edits)
- **Oracle labels (OracleRRI)**: candidate poses → metric depth renders from GT mesh → backproject to world points → fuse with current semidense reconstruction → compute point↔mesh Chamfer components → RRI labels.
- **VIN v2 (VinModelV2)**: frozen EVL backbone → compact voxel scene field (`occ_pr`, `occ_input`, `counts_norm`, `cent_pr`, `free_input`, `new_surface_prior`) → pose encoding (`t + R6D` → LFF) → pose-conditioned global pooling (MH cross-attn over coarse voxel tokens).
- **View conditioning beyond local voxels**: project semidense points into each candidate camera using PyTorch3D `PerspectiveCameras.transform_points_screen` → (i) scalar projection stats (`coverage`, `empty_frac`, `valid_frac`, `depth_mean`, `depth_std`) and (ii) per-point tokens (`x_norm`, `y_norm`, `depth_m`, `inv_dist_std`) aggregated with MHCA (pose queries).
- **Training objective**: CORAL ordinal thresholds with correct cumulative→marginal conversion; optional balanced/focal threshold loss variants; auxiliary Huber on expected continuous value derived from marginal probs and monotone bin representatives.

## Section-by-section review (paper correctness + fidelity)

### 1) Intro / Related work
- Reads cleanly and is consistent with the implementation and the VIN-NBV framing.
- Abstract uses multiple acronyms (NBV/RRI/ASE/EVL/EFM3D) IEEE-style abstracts typically avoid; acceptable for an internal draft.
- Related work is focused (VIN-NBV, GenNBV, EFM3D/EVL, SceneScript). Classical NBV citations are intentionally not expanded; add if targeting a broader venue.

### 2) Problem formulation
- RRI definition and Chamfer components are coherent and match the oracle pipeline conceptually.
- Consider adding a short sentence clarifying how the implementation approximates mesh integrals (surface sampling) and how accuracy vs completeness are logged.

### 3) Dataset and inputs
- Correctly states ASE scale (100k scenes) and local `.data/ase_efm` snapshot (100 GT-mesh scenes, 4,608 snippets; median 40).
- The 100 vs 1,641 GT-mesh figures are aligned with `docs/contents/ase_dataset.qmd`; the paper already calls out the *current snapshot* vs the advertised larger subset.

### 4) Coordinate conventions
- Correct, but could mention the practical `rotate_yaw_cw90` “UI alignment” convention and that VIN v2 undoes it before feature computation (since it is a recurring source of confusion in diagnostics).

### 5) Oracle RRI computation
- Good separation: paper makes it explicit that VIN does not access GT meshes/renders directly.
- Pipeline wiring summary aligns with `OracleRriLabeler` + candidate generation + depth renderer + pointcloud fusion + Chamfer/RRI modules.

### 6) VIN v2 architecture
- Scene-field channels and derived features are implementation-aligned (`new_surface_prior = (1 - counts_norm) * occ_pr`).
- Semidense projection description now matches the current code path (PyTorch3D `transform_points_screen`, not `CameraTW.project`).
- Frustum context description matches the current MHCA over projected semidense point tokens.
- Optional PointNeXt + trajectory components are described as optional, consistent with config-as-factory patterns.

### 7) Training objective (CORAL)
- Correctly calls out the main conceptual pitfall: CORAL produces cumulative threshold probabilities.
- Uses the correct cumulative→marginal conversion before expectations; consistent with `docs/contents/impl/coral_intergarion.qmd` and the code.
- Notes balanced/focal threshold losses as an escalation for collapse/imbalance.

### 8) Evaluation / ablations / WandB analysis
- Evaluation metrics are consistent with the ATEK surface reconstruction protocol.
- Ablation and W&B tables are now readable in the two-column layout (reduced font size + narrower columns).
- W&B analysis is honest about NaN runs and the small sample size (only runs with >500 steps).

### 9) Discussion / conclusion / appendix
- Conclusion is placed before appendices (better narrative flow than having it after the gallery).
- Appendix contains the heavy diagnostics screenshots (as requested) and keeps the main text cleaner.

## Layout + readability issues (remaining)
- The “main body” ends close to the appendix start; consider a `#pagebreak()` before the appendix include for a cleaner separation.
- Some large screenshots in the appendix are still dense; if targeting a strict page limit, prune to the most diagnostic figures or combine into montages.

## Suggested next edits (high value)
1) Add a single sentence in the coordinate section clarifying `rotate_yaw_cw90` and when it is applied/undone.
2) Add a short “implementation note” in the RRI section linking the theoretical Chamfer to the batched implementation (accuracy/completeness terms and sampling).
3) If you want the paper to be 15 pages: prune 1 appendix page (e.g., move two histograms out, or reduce screenshot heights).

## Status update (applied in this iteration)
- Added the `rotate_yaw_cw90` UI-alignment note and clarified that VIN v2 undoes it before feature computation (Coordinate Conventions section).
- Added an implementation note linking the theoretical Chamfer split to the PyTorch3D point↔triangle and triangle↔point operators used in code (Oracle RRI section).
- Pruned the appendix gallery to meet the 15-page target (removed the last four jitter-related figures) and recompiled the paper.

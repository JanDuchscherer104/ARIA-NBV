# EFM3D related-work review (2026-01-07)

## Task summary
- Review `docs/typst/paper/sections/02-related-work.typ` (Egocentric foundation models subsection).
- Cross-check against `docs/contents/literature/efm3d.qmd` and EFM3D LaTeX sources in `literature/tex-src/arXiv-EFM3D/`.
- Provide missing points and flag any incorrect claims.
- Provide overview of paper contents (`docs/typst/paper/main.typ`).

## Files reviewed
- `docs/typst/paper/main.typ`
- `docs/typst/paper/sections/02-related-work.typ`
- `docs/contents/literature/efm3d.qmd`
- `literature/tex-src/arXiv-EFM3D/main.tex`
- `literature/tex-src/arXiv-EFM3D/abstract.tex`
- `literature/tex-src/arXiv-EFM3D/intro.tex`
- `literature/tex-src/arXiv-EFM3D/related.tex`
- `literature/tex-src/arXiv-EFM3D/dataset.tex`
- `literature/tex-src/arXiv-EFM3D/method.tex`

## Paper contents overview (from `main.typ`)
- Introduction
- Related Work (NBV, EFM3D/EVL, SceneScript, CORAL, FiLM)
- Problem Formulation (Chamfer/RRI, ordinal binning)
- Dataset and Inputs
- Coordinate Conventions and Geometry
- Oracle RRI Computation
- Aria-VIN-NBV Architecture (baseline + ablations)
- Semi-dense Frustum Pooling & view-conditioned tokens
- Training Objective (CORAL, auxiliary regression, balancing)
- Stage-aware binning + priors
- Training configuration snapshot
- System pipeline (candidate generation, rendering, diagnostics)
- Diagnostics + preliminary findings
- Evaluation protocol
- Ablation plan
- WandB run analysis
- Discussion & limitations
- Toward entity-aware NBV
- Conclusion
- Appendices: OracleRriLabeler pipeline, VIN v2 implementation notes, candidate sampling gallery, extra diagnostics

## Findings: Egocentric foundation models subsection

### Correct/accurate statements
- EFM3D is a benchmark + model stack for egocentric 3D perception.
- EVL lifts multi-stream/time observations into a local voxel grid.
- EVL outputs occupancy and OBB-related signals including centerness.

### Missing details worth adding (if space allows)
- **Tasks/benchmark scope**: EFM3D explicitly benchmarks **3D OBB detection and 3D surface regression** on egocentric Aria data (abstract + intro).
- **Input modalities**: EVL uses posed/calibrated **RGB + grayscale SLAM streams** and **semi-dense SLAM points** with **visibility-derived surface/free-space masks** (method + intro).
- **Backbone + architecture**: frozen **2D foundation model features** (DINOv2.5 in the paper) lifted into a **gravity-aligned voxel grid** anchored to the last RGB pose, then processed by a **3D U-Net** and task heads (method).
- **Local grid contract**: EVL’s grid is **local (e.g., 4 m^3)** and gravity-aligned; this matters for NBV candidate coverage and motivates out-of-bounds handling.
- **Dataset contributions**: EFM3D adds ASE OBBs/visibility metadata and GT meshes for ASE validation + ADT; introduces a small real-world OBB set (AEO) for sim-to-real evaluation (dataset/intro).

### Potential clarifications (not errors)
- “Multi-view” can be interpreted as **multi-stream + multi-time**; consider clarifying that EVL aggregates features across time and streams rather than true multi-camera simultaneous views.
- If you mention DINOv2.5 or the 3D U-Net in related work, add the corresponding citations (DINOv2/ViT-Need-Reg, etc.).

### No incorrect claims found
- The current paragraph is consistent with EFM3D/EVL descriptions in the LaTeX sources; it is simply minimal.

## Suggested optional wording (one-sentence expansion)
- “EFM3D benchmarks **3D OBB detection and surface regression** on egocentric Aria data and introduces **EVL**, which lifts multi-stream video features (RGB + SLAM) and semi-dense point/free-space masks into a **gravity-aligned local voxel grid** processed by a 3D U-Net.”

## Open suggestions for future edits
- If space permits, add one sentence linking EVL’s **local voxel extent** to our NBV motivation for semidense projection features (out-of-bounds candidates).
- Ensure any new citations are present in `docs/references.bib`.

## Update applied
- Expanded the EFM3D paragraph in `docs/typst/paper/sections/02-related-work.typ` to cover task scope, input modalities, local grid contract, architecture, and dataset additions, all cited to `@EFM3D-straub2024`.
- Added a short NBV-specific link to EVL’s local extent motivating semidense projection cues for out-of-bounds candidates.
- Added `@UNet3D-cicek2016` to `docs/references.bib` and cited it for the 3D U-Net mention.
- Synced `docs/_shared/references.bib` with `docs/references.bib` after adding the new entry.

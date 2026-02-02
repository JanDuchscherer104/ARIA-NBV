# Task Notes: Oracle RRI Labeler MMDC Flowchart (2026-01-26)

## Scope
- Created Mermaid (mmdc) flowchart iterations for `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.
- Iterations are kept as separate `.mmd` + `.png` files for comparison.

## Outputs
- `external/mmdc-examples/oracle-rri-labeler-v1.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v1.png`
- `external/mmdc-examples/oracle-rri-labeler-v2.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v2.png`
- `external/mmdc-examples/oracle-rri-labeler-v3.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v3.png`
- `external/mmdc-examples/oracle-rri-labeler-v4.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v4.png`
- `external/mmdc-examples/oracle-rri-labeler-v5.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v5.png`
- `external/mmdc-examples/oracle-rri-labeler-v6.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v6.png`
- `external/mmdc-examples/oracle-rri-labeler-v7.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v7.png`
- `external/mmdc-examples/oracle-rri-labeler-v8.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v8.png`
- `external/mmdc-examples/oracle-rri-labeler-v9.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v9.png`
- `external/mmdc-examples/oracle-rri-labeler-v10.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v10.png`
- `external/mmdc-examples/oracle-rri-labeler-v11.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v12.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v13.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v13.png`
- `external/mmdc-examples/oracle-rri-labeler-v14.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v14.png`
- `external/mmdc-examples/oracle-rri-labeler-v15.mmd`
- `external/mmdc-examples/oracle-rri-labeler-v15.png`

## Diagram Coverage
- End-to-end pipeline: snippet -> candidate generation -> depth rendering -> backprojection -> oracle RRI -> batch.
- Candidate generation internals: PositionSampler, OrientationBuilder, pruning rules.
- Rendering internals: CandidateDepthRenderer -> Pytorch3DDepthRenderer.
- Backprojection internals: `_backproject_depths_p3d_batch` -> `build_candidate_pointclouds`.
- Oracle internals: `OracleRRI` -> Chamfer metrics.

## Notes
- v4 switches to a top-to-bottom layout with larger rank spacing to reduce horizontal compression.
- All labels now use `<br/>` for line breaks (no `\n` literals), so MMDC renders multi-line text correctly.
- Colors and rounded subgraph styling mirror `external/mmdc-examples/doc-classifier-config.mmd`.
- mmdc requires elevated permissions to launch the browser sandbox on this system.

## Tests
- `mmdc` used to render PNGs (requires escalated permissions).

## Iteration v6
- Added node classes to distinguish compute (orange), data (blue), and result (green).
- Output: `external/mmdc-examples/oracle-rri-labeler-v6.mmd` + `.png`.

## Iterations v7-v15
- v7: attempted KaTeX-style multiline labels; fixed by moving to array blocks per label.
- v8: stable multiline math labels.
- v9: compact layout variant (ELK) edited manually.
- v10: first pass with symbolic edge labels; parse failures when using `|$$...$$|`.
- v11: KaTeX parse errors due to multiple `$$...$$` blocks in a label.
- v12: switched to a single math block per label using `\\begin{array}{c}`.
- v13: consistent `\\texttt{...}` wrapping for all class/function names; uses `\\mathcal{}`/`\\mathbf{}` symbols and renders cleanly.
- v14: LR layout with pruning mask combine/apply steps and tensor shapes (e.g., `\\mathcal{Q} \\in \\mathrm{SE}(3)^{N_q}`).
- v15: larger fonts and `.title`-styled cluster titles; tighter spacing; TB layout for compact height.

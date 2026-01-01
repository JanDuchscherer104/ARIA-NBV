# slides_3.typ — Oracle RRI pipeline slides

## Goal

Revise `docs/typst/slides/slides_3.typ` into a stage-by-stage, theory + implementation overview of the `oracle_rri` *oracle label* pipeline:

`data handling → candidate generation → depth rendering → backprojection → oracle RRI metric`

Starting point: `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.

## What changed

- Rewrote `docs/typst/slides/slides_3.typ` to focus on the oracle-RRI pipeline stages and the conceptual background behind each stage.
- Added slides for missing pipeline stages:
  - Depth rendering (PyTorch3D rasterization vs. CPU raycasting fallback).
  - Backprojection/unprojection (pixel→NDC mapping and batched unprojection).
  - Oracle RRI computation (RRI definition + accuracy/completeness point↔mesh terms).
  - A practical breakdown of `chamfer_point_mesh_batched` (vectorization strategy shown as pseudocode).
- Updated math notation to consistently use:
  - bold symbols for vectors/matrices/tensors (e.g. `bold(p)`, `bold(R)`),
  - bold calligraphic for point clouds and meshes (e.g. `bold(cal(P))`, `bold(cal(M))`),
  - split candidate generation into positional vs directional slides,
  - face-set notation as `bold(cal(F))` with elements `bold(f)`,
  - explicit denominators like `1/(|F|)` so both bars render reliably.
- Added a “Resolved Issues (High Impact)” slide summarizing the major completed items from `docs/contents/todos.qmd` (refactor/caching/indexing/stage consolidation/sampling determinism).
- Added a condensed “Rendering & Backprojection: Issues and Fixes” slide sourced from the “Previously observed issues” section in `docs/contents/todos.qmd`.
- Ensured Typst syntax is valid (avoid LaTeX commands like `\quad`, use Typst-native quoting for multi-letter subscripts, keep code-like expressions in raw blocks).
- Recovered the deck after an accidental `git restore` on `slides_3.typ` by fully rewriting the file from the latest intended structure.
  - **Typst gotcha**: math line breaks use a single backslash `\` (not LaTeX `\\`), otherwise the line-break will render as a literal `\`.

## How to compile

Typst file imports and bibliography use root-relative paths under `docs/`, so compile with:

```bash
typst compile --root docs docs/typst/slides/slides_3.typ docs/typst/slides/slides_3.pdf
```

## Open follow-ups / suggestions

- **Depth validity + histogram semantics:** In PyTorch3D renders, miss pixels often show `pix_to_face=-1` and can show `z=-1`; plots should always apply the valid mask before computing depth statistics.
- **Mesh watertightness:** “See-through” behavior is often caused by open meshes (especially after cropping). The CPU renderer has a proxy-wall sealing option; consider a similar strategy (or a stronger cropping heuristic) for the PyTorch3D path if this remains an issue.
- **Density matching for RRI:** `OracleRRIConfig.voxel_size_m` exists to equalize densities between `P_t` and `P_{t∪q}`; if RRI becomes overly sensitive to candidate point counts, a consistent voxel-downsampling step is the next lever.
- **Cropping to occupancy bounds:** `OracleRRI` currently keeps full mesh tensors (cropping helper exists but is commented out). If compute becomes heavy or multi-room meshes bias distances, cropping to `CandidatePointClouds.occupancy_bounds` should be reconsidered.

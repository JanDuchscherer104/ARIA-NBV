# chamfer_point_mesh_batched (slides_3.typ) — notes

## What was added

Added a theory/conceptual breakdown of `oracle_rri/oracle_rri/rri_metrics/metrics.py::chamfer_point_mesh_batched` to `docs/typst/slides/slides_3.typ` (new slides around pages ~21–23 in the compiled PDF).

Changes included:

- Added 3 slides explaining inputs/outputs, formulas (accuracy/completeness/bidirectional), and the packed/vectorised batching strategy.
- Switched notation to `cal(P)` / `cal(M)` and made indices explicit (`i` candidate, `k` point, `j` triangle, `N_i = l_i`).
- Added hyperlinks to the external PyTorch3D implementation:
  - `point_mesh_distance.py` (main branch)
  - `point_face_distance`, `face_point_distance`, and `_DEFAULT_MIN_TRIANGLE_AREA` anchors.

Note: `docs/typst/slides/slides_3.typ` contained unrelated syntax issues in the later “Candidate View Generator / Rules” section; these were fixed minimally to satisfy the “slides compile” constraint.

## Key implementation facts (matches code)

`chamfer_point_mesh_batched(points, lengths, gt_verts, gt_faces)` computes **per-candidate** point↔mesh distance components:

- **Accuracy** (point → mesh): mean over points of the **squared** distance to the closest triangle.
- **Completeness** (mesh → point): mean over triangles of the **squared** distance to the closest point.
- **Bidirectional**: sum of the two.

It is fully vectorised across the candidate batch dimension `C`:

- Pads → mask → **pack points** into `(P_tot, 3)`.
- Repeats the GT mesh `C` times to align with PyTorch3D’s packed batch API.
- Uses CUDA kernels:
  - `pytorch3d.loss.point_mesh_distance.point_face_distance` for P→M.
  - `pytorch3d.loss.point_mesh_distance.face_point_distance` for M→P.
- Converts sums to means via per-point/per-triangle weights and aggregates with `scatter_add_`.

## Metric interpretation caveats (worth remembering)

- Distances are **squared L2** (units: m² if inputs are in meters). If you want L2 distances in meters, you’d use `sqrt(d^2 + eps)`; the current oracle code intentionally avoids the sqrt.
- Completeness is averaged **per triangle**, not area-weighted and not via mesh surface sampling. This is fast and differentiable, but can overweight regions with many small triangles.

## How to (re-)compile locally

Typst needs the docs folder as root because slides import `docs/typst/shared/macros.typ` and read figures from `docs/figures/`:

```bash
typst compile --root docs docs/typst/slides/slides_3.typ docs/typst/slides/slides_3.pdf
```

## Potential follow-ups (not implemented)

- Add an optional **area-weighted** completeness variant (triangle areas as weights) if we want closer alignment with surface-sampling-based metrics.
- Add a “sqrt vs squared” switch in metrics and clearly propagate it to RRI normalization to avoid unit confusion when comparing across experiments.

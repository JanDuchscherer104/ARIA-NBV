# oracle_rri pipeline-stage slides (slides_3.typ)

Goal: Extend `docs/typst/slides/slides_3.typ` with **theory + conceptual background** for the *other stages* of the `oracle_rri` pipeline (candidate generation, depth rendering, backprojection/fusion), using `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py` as the stage-graph reference.

## What changed

- Added a compact pipeline “stage graph” slide that mirrors `OracleRriLabeler.run`:
  - candidate generation → depth rendering → backprojection+fusion → scoring.
- Added conceptual slides for:
  - Candidate view generation (sampling + pruning rules)
  - Orientation construction (look-at frames + view jitter)
  - Rendering GT mesh → depth (`Pytorch3DDepthRenderer`)
  - CandidateDepthRenderer oversample + hit-count filtering
  - Backprojection depth → candidate point clouds (`build_candidate_pointclouds`)
  - Fusion of semi-dense SLAM points with candidate points for scoring
- Fixed notation to consistently use calligraphic sets `cal(M)`, `cal(P)` and consistent indices (`i` for candidate, `(u,v)` for pixels, etc.).
- Replaced fragile footnotes in the rendering/backprojection slides with **inline GitHub links** to avoid accidental page breaks/blank slides.
- Minor math/layout fixes:
  - Avoided math subscripts like `T_{world<-cam}` being parsed as undefined variables by using `T_("world" arrow.l "cam")`.
  - Replaced line-break hacks (`\\`) with `#linebreak()` where needed.

## External source links used (valid)

- PyTorch3D:
  - `MeshRasterizer`: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/rasterizer.py#L139
  - `PerspectiveCameras`: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L1030
  - `PerspectiveCameras.unproject_points`: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L1154
  - point↔mesh distance kernels: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/point_mesh_distance.py
- EFM3D tensor wrappers:
  - `PoseTW`: https://github.com/facebookresearch/efm3d/blob/main/efm3d/aria/pose.py
  - `CameraTW`: https://github.com/facebookresearch/efm3d/blob/main/efm3d/aria/camera.py
- `power_spherical` (orientation forward-bias alternative):
  - https://github.com/andife/power_spherical

## Build / verification

- Re-generated the deck:
  - `typst compile --root docs docs/typst/slides/slides_3.typ docs/typst/slides/slides_3.pdf`
- Spot-checked rendering around the new section:
  - `pdftoppm -f 41 -l 46 -png -r 150 docs/typst/slides/slides_3.pdf /tmp/slides_3_check`

## Notes / caveats

- Coordinate conventions:
  - `PoseTW` is column-vector SE(3); PyTorch3D uses a row-vector convention for `PerspectiveCameras` (`X_cam = X_world R + T`). The renderer therefore passes `R^T` when constructing PyTorch3D cameras (documented on-slide).
- Depth semantics:
  - The slides assume metric depth (`in_ndc: false`) and explain validity as `(finite + z-range + hit mask pix_to_face>=0)`.

## Files touched

- `docs/typst/slides/slides_3.typ`
- `docs/typst/slides/slides_3.pdf` (rebuilt)


# Depth mesh debug toggle

- Added optional mesh plotting in the depth-hit backprojection panel: when the checkbox is enabled, a trimesh is constructed from `sample.mesh_verts`/`sample.mesh_faces` (if available) and fed into `RenderingPlotBuilder.from_snippet(...)`.
- Leaves the sample mesh untouched when already present; warns if verts/faces are missing.


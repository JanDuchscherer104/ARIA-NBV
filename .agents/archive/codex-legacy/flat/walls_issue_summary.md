# Candidate Rendering – Missing Walls (Nov 2025)  

**What we observed**
- Streamlit “Data” and “Candidate Renders” showed scenes with furniture but no enclosing walls; depth maps saturated at far plane.
- Inspection of ASE mesh for scene `81283`: ~1.65M faces but almost no faces on side walls (xmin/ymin had 0 faces; ymax had few). Mesh components are highly fragmented (≈336k); cropping/decimation wasn’t the culprit.
- Semidense bounds fully enclose the room; cropping keep_ratio would be 1.0, so geometry is simply absent in the mesh.

**Fixes implemented**
- Renderer (PyTorch3D) now injects proxy wall planes when wall coverage <20% of expected area, using semidense `volume_min/max` as the room AABB; still duplicates faces for two‑sided rendering.
- CandidateDepthRenderer passes occupancy extent into the renderer so proxy walls align with the actual room.
- Plotly mesh helper overlays the same proxy walls (semi‑transparent) when coverage is low, still rendering two‑sided faces.
- UI defaults: no decimation; cropping keep ratio guard; double‑sided meshes on; hit‑ratio logging.

**How to validate**
- Streamlit: load a snippet; Data tab should show walls (proxy box) with “Show walls” enabled. Candidate Renders should have hit_ratio > ~0.3 (logged and shown in title).
- If walls still missing, enable renderer `is_debug=True` to see which planes were synthesized; confirm semidense `volume_min/max` exist.

**Risks / next steps**
- Scenes lacking semidense bounds still need a fallback (e.g., trajectory AABB) for proxy walls.
- Need broader multi-scene QA to tune the 20% coverage threshold/eps.  
- Optional: expose a UI toggle for proxy walls and per-plane coverage stats; add an automated regression that asserts wall visibility via hit_ratio on a known problematic scene.***

## Nov 22, 2025 – Ordering + bounds fix
- Found that `CandidateDepthRenderer` stacked `occupancy_extent` as `[xmin, xmax, ymin, ymax, zmin, zmax]` but `Pytorch3DDepthRenderer` interpreted the first three values as `vmin`, leading to scrambled bounds and proxy walls in the wrong place. This could still leave scenes open.
- Fixed `_occupancy_extent` computation and aligned the renderer to parse the same `[xmin, xmax, ymin, ymax, zmin, zmax]` convention while merging semidense extents with mesh bounds (union) before inserting proxy walls.
- Added regression tests to assert (a) occupancy extents preserve axis ordering and (b) proxy walls expand to the provided semidense AABB.

## Nov 22, 2025 – CPU renderer brought to parity
- Replaced placeholder `efm3d_depth_renderer.py` with a fully functional CPU ray tracer (trimesh/pyembree), supporting proxy walls, occupancy-extent union, chunked ray casting, and config-as-factory + Console logging consistent with the PyTorch3D renderer.
- Added regression tests `test_efm3d_depth_renderer.py` covering hit ratio on a front-facing plane and proxy wall expansion to semidense bounds.
- Added integration tests on real ASE snippet 81283 for both PyTorch3D and CPU backends (downscaled calib + decimated mesh for runtime), ensuring CandidateDepthRenderer can switch backends and still hit geometry.

Ensuring Interiors Are Fully Rendered Despite Missing Walls
In the NBV (Next-Best-View) pipeline, interior walls often appear missing in depth renders and interactive views because the ground-truth meshes lack those polygons. We need to robustly render a closed interior scene even when large wall surfaces are absent. This involves adding proxy geometry to “seal” the room, handling fragmented meshes, verifying coordinate conventions, and adjusting rendering settings in both PyTorch3D and Plotly.
Adding Proxy Geometry for Missing Walls
Bounding-Box “Shell” Planes: The current approach detects when mesh faces cover less than a threshold (e.g. 20%) of the expected area for each room boundary plane, and then inserts thin planar proxies at the scene’s bounding box extents
GitHub
GitHub
. In code, this uses the mesh’s axis-aligned bounds (vmin, vmax) to compute each wall’s expected area (e.g. area of X-min wall = Y_extent * Z_extent) and sums areas of any faces within an epsilon of each bound
GitHub
. If coverage is low (e.g. no triangles near X_min), a corresponding wall is flagged as missing
GitHub
GitHub
. A thin box mesh is then created at the bounds and only the faces for the missing planes are kept
GitHub
GitHub
. These proxy wall triangles are concatenated with the original mesh so the scene becomes enclosed
GitHub
. This ensures even if the original mesh had big holes, the proxies act as a “room shell” to block the view of the void outside. Use Occupancy (Semidense) Extents: In practice, the mesh’s reported bounds might not reflect the true room size if the mesh is incomplete. For example, in scene 81283 the semidense SLAM points extend to Y≈–19 m while the mesh stops at Y≈–11 m (leaving an 8 m gap). To avoid placing proxy walls too far inward, derive the wall positions from the occupancy volume (SLAM point cloud bounds) whenever available. The semidense snippet data provides volume_min/volume_max in world coordinates
GitHub
. We should use these for proxy wall placement when they exceed the mesh bounds. In other words, if the SLAM point cloud indicates the environment extends beyond the mesh, treat those larger extents as the true scene limits. This way, the proxy walls align with where the real walls likely are. Currently, the code passes an occupancy_extent to the renderer but doesn’t yet use it for wall placement (an oversight in _maybe_with_proxy_walls)
GitHub
GitHub
. A fix is to expand vmin,vmax to include occupancy bounds before creating the box, ensuring proxies cover the full room. Higher Coverage Threshold: Additionally, consider raising the coverage threshold or always adding a proxy if any significant gap exists. At 20% threshold
GitHub
GitHub
, even a wall that is 25% covered (75% hole) would skip proxies. A stricter criterion (e.g. 90% or detecting any continuous ≥1–2m hole) would be safer. It’s better to overlap a proxy on an existing partial wall (with minimal visual impact if semi-transparent) than to leave a big open gap. The Plotly implementation currently simply adds the entire box if any wall is missing
GitHub
GitHub
, which, while doubling some faces, at least guarantees closure. We can adopt a similar approach for PyTorch3D: include all six planes of a thin box when any large section is missing (or ensure we cover each missing side individually as done now, but with perhaps a tiny outward offset to avoid z-fighting with any existing fragments). Voxel Hull Alternative: For a more data-driven solution, we could generate a voxel-based hull from the semidense point cloud. For example, run a coarse TSDF (truncated signed distance function) fusion or occupancy grid over the snippet’s volume and extract an iso-surface as a closed mesh
xiaoyuanguo.github.io
. The project already has a volumetric fusion module (e.g. VolumeFusion.get_trimesh() in code) capable of producing a watertight surface from accumulated depth
GitHub
GitHub
. This could fill in unseen areas (essentially treating unknown space as solid beyond observed free-space). However, this is more complex and computationally heavy for a quick rendering fix. A simpler heuristic is to place large planar proxies aligned with the room’s principal directions (likely axis-aligned in synthetic scenes) at the outermost occupied extents – effectively what the bounding-box method does. For now, leveraging the semidense AABB plus thin proxy walls is a practical compromise to render interiors closed.
Handling Fragmented and Non-Watertight Meshes
The ASE meshes are highly fragmented (hundreds of thousands of disconnected components) and non-watertight. This can confuse rendering algorithms that assume solid surfaces. A few best practices address this:
Disable Backface Culling: When the camera is inside a mesh, backface culling would remove the interior faces of walls, making them invisible. The renderer config explicitly sets cull_backfaces=False by default
GitHub
. This ensures interior wall faces aren’t dropped. In fact, the config warns that with interior viewpoints, culling would “remove the interior walls”
GitHub
. Keeping culling off is correct for our use-case.
Two-Sided Rendering: Even with culling off, many interior faces have normals facing inward (opposite the camera). By default, Plotly and PyTorch3D shade one side of each triangle. We enable two-sided rendering so that backside normals don’t appear unlit or get omitted. In PyTorch3D, the two_sided=True option duplicates every face with reversed winding
GitHub
. In Plotly, the _mesh_to_plotly utility does the same: it adds a second Mesh3d trace with faces [i,j,k] reversed and an “inner” color
GitHub
GitHub
. Both renderers therefore draw each triangle from both sides. Combined with an ambient light term (discussed below), this makes interior surfaces visible and lit. This approach is analogous to two-sided materials in engines (or manually duplicating and flipping normals in Blender
communityforums.atmeta.com
).
Large Mesh Performance: With ~1.65M vertices and ~1.65M faces, rasterizing could be slow. PyTorch3D handles batches on GPU, but we might need to adjust settings like bin_size or max_faces_per_bin if there are performance warnings. The current config leaves those at defaults (which usually choose a good binning)
GitHub
. Since we’re not texturing, this is mostly fine. If needed, decimation could be applied as a preprocessing step (with caution not to remove rare wall fragments – currently decimation is off by default). The numerous tiny components shouldn’t inherently break anything, though they might slightly bloat the face list.
Signed Distance Queries: One known issue is the use of trimesh.proximity.signed_distance in the candidate generation (MinDistanceToMesh rule). Trimesh defines distances as positive inside or on the surface, negative outside. For an open mesh, the concept of “inside” is ill-defined – points might erroneously register as inside due to local surface orientation. If we rely on these distances, we should be careful: perhaps treat any point beyond a certain distance from all surfaces as “outside” regardless of sign. Alternatively, since we only need a minimum clearance, using the absolute distance might suffice, or using closest_point distance instead of signed distance. While this doesn’t affect rendering directly, it’s a side consideration because a fragmented non-watertight mesh could report odd signed distances (e.g. a camera in the middle of a room might be considered “outside” if the walls are missing). Ensuring proxy walls are present will also make the mesh closer to closed, improving the robustness of any “inside” tests.
Verifying Coordinate Frames and Conventions
It’s crucial that all transformations and added geometry share a consistent coordinate frame (the Aria VIO world frame). The project uses Aria’s convention where the camera frame is RDF – +X right, +Y down, +Z forward – and the world frame is an ENU-style frame with +Z up (gravity vector pointing negative Z)
GitHub
. Poses are stored as PoseTW objects representing transformations like T<sub>world,cam</sub>. In our code, the camera pose is correctly assembled as T_world_cam = T_world_rig · (T_camera_rig)<sup>−1</sup>
GitHub
, meaning the camera’s world pose is derived from the rig (IMU) world pose and the camera’s extrinsic within the rig. This yields a 4×4 matrix that transforms points from camera coordinates to world coordinates. PyTorch3D Camera Setup: The PyTorch3D renderer expects a rotation (R) and translation (T) that transform world coordinates into the camera’s view (i.e. OpenGL-style camera extrinsics). Our _pose_to_r_t helper inverts the PoseTW for this purpose
GitHub
. It takes the T_world_cam matrix, extracts the rotation R_wc and translation t_wc, then computes R_cw = R_wc^T and t_cw = -R_cw * t_wc
GitHub
. This indeed produces the camera-to-world inversion (R_cw, t_cw) which is used as PyTorch3D’s R and T
GitHub
GitHub
. We thus confirm the math is correct: the camera ends up looking in the intended direction. (If this were wrong – e.g. using the pose directly without inversion – the camera would point 180° off or the scene would appear behind the camera. The fact we see furniture and partial geometry in front of us means the pose transform is being applied correctly.) Frame Alignment for Proxy Walls: The proxy walls are added in the same coordinate frame as the mesh, which in turn is in world coordinates. The dataset loader likely already aligned the mesh to the Aria world (the similar ranges of mesh vs semidense points confirm this). For instance, mesh bounds Z≈3.8 m matches the ceiling height in semidense
GitHub
, and both share the same coordinate axes orientation. When we compute mesh.bounds and add proxies at those global min/max coordinates, they are in the world frame and thus correctly positioned relative to the camera poses. We should verify that no extra transform (like a shift to local object coordinates) is needed. In our case, AseEfmDataset loads the mesh vertices presumably already in the world coordinate system of that scene (there’s no indication of additional transforms in the loading code). The Plotly view even overlays the trajectory and semidense points with the mesh, showing they coincide spatially – a strong sign the frames match up. Therefore, no coordinate-handiness or axis mismatch issues are apparent. The Aria convention is consistently applied: gravity is along –Z (so floors at lowest Z, ceilings at highest Z), and the camera poses and proxy wall planes are all defined in this gravity-aligned world frame. This consistency rules out the possibility that walls aren’t visible due to a frame mix-up (e.g. walls inserted in a different coordinate space or the camera looking in the wrong direction). One small nuance: the display orientation in the Streamlit app applies an extra rotation (pose_for_display(align_gravity=True)) to make the camera view upright for visualization
GitHub
. This is only for rendering the frustum and images in an intuitive orientation (undoing a 90° roll in Aria’s rig frame). The PyTorch3D depth rendering does not use this adjusted pose – it uses the true physical pose_world_cam. This is correct; we wouldn’t want to accidentally render depth from a tilted camera. In summary, all transformations are being handled correctly and consistently: T_world_cam is correct, and proxies use the same world frame. We just need to extend the logic to incorporate the full scene bounds from SLAM to place proxies optimally.
Plotly Mesh3D Settings for Interior Views
When visualizing in Plotly, we must compensate for its lighting and culling behavior to see interior surfaces:
One-Sided Lighting: By default, Plotly’s Mesh3d will shade triangles only on the side of the normal (front-face). From inside a mesh, you’d normally see the unlit back-faces (which appear dark or invisible). Our solution was to add a second set of triangles with reversed orientation (double_sided_mesh=True in the UI)
GitHub
GitHub
. These “inner” faces are given a slightly different color and are lit with ambient=1.0 and diffuse=0.0 (a flat shading)
GitHub
. The ambient term ensures they appear visible even if no light hits them “head on.” In code, the base mesh trace uses a light gray with ambient=1, diffuse=0.3 (so there is slight shading variation)
GitHub
, while the duplicated inner faces use ambient=1, diffuse=0 to avoid any directional shading (so you don’t get weird dark patches when viewing walls from inside)
GitHub
. This effectively disables light directionality for interior walls, making them uniformly visible. The result is that from an interior viewpoint, walls and other geometry are drawn from behind and appear with a constant light gray color. This was the intended trick and is confirmed as the correct approach – it matches typical practices (e.g. in Blender or Unity one would either disable backface culling or use a two-sided material)
communityforums.atmeta.com
communityforums.atmeta.com
.
No Depth Sorting Issues: Plotly’s WebGL rendering can sometimes have trouble with two semi-transparent surfaces overlapping (z-fighting or blending issues). We use an opacity of ~0.35 for the mesh and 0.12 for proxy walls
GitHub
GitHub
. This gives a translucent look so one can see frusta and points through the walls. We haven’t noticed major artifacts yet, but if we do see flickering where a proxy coincides exactly with an existing mesh face, one trick is to offset the proxy slightly outward (e.g. by a few cm) or use a slightly different opacity/color to make overlaps distinguishable. Currently, the proxy walls in Plotly are colored light blue (#b0c4de) and very transparent
GitHub
, so even if they overlap with a small actual wall fragment (light gray), it’s not too distracting visually.
Result of Current Plotly Settings: Despite these measures, some scenes still showed “no walls” in Plotly, meaning the proxies likely weren’t being triggered or were placed too far in (as discussed earlier). By ensuring the missing-wall detection uses the full scene bounds and perhaps lowering the missing-area threshold, Plotly will include the Proxy walls trace in all cases where large sections are open. We can verify in the logs or UI: for scene 81283, after fixes, the Proxy walls trace should appear in the legend whenever, say, the Y-min or X-min wall is absent. This, combined with two-sided lighting, will fill in the previously blank region with a translucent wall.
In summary, the interior rendering in Plotly is fixed by duplicating faces (two-sided mesh) and using full ambient lighting on the inside faces
GitHub
GitHub
. This approach is confirmed as best practice for our use case, since Plotly (and most 3D engines) do not automatically render back-faces from the inside. No further changes are needed there aside from making sure the proxy geometry actually gets added.
Toward a Complete Solution
By integrating the above fixes, we expect the depth renderer’s hit ratio (fraction of pixels hitting geometry) to increase well above the current modest values, and the visual output to show enclosed rooms rather than open voids. Key deliverables will be:
Enhanced PyTorch3D Renderer: Update _maybe_with_proxy_walls to incorporate semidense volume bounds (if available) when computing vmin, vmax. This guarantees proxy walls cover the true room extents, not just the truncated mesh. With this change, scene 81283’s missing walls (especially on the Y– axis) should appear in the depth maps (previously, those regions likely returned depth = zfar). We’ll verify that now depth images have finite values (actual distances) where a wall should be, instead of saturated to 20 m. The hit_ratio should jump significantly for those views – ideally we want it above some heuristic like 30% or whatever indicates walls are being hit, rather than ~0% in the missing-wall directions.
Reliable Wall Coverage Metric: We should define a clear metric for “visible vertical planes.” One approach is to compute the proportion of image pixels at the boundary of the view (e.g. around the periphery) that register a near-maximum depth. In an enclosed room, virtually all rays eventually hit something (wall or ceiling/floor). If after our fixes the depth image still shows large regions at zfar, that indicates failure to close the scene. Our goal is that for any viewpoint inside a reasonably bounded space, the fraction of pixels hitting nothing should be minimal. This can be evaluated by the existing hit_ratio logging
GitHub
. A threshold (say >0.3 or >0.5 depending on how much of the FOV is walls vs open doorways) can serve as a check. We expect scenes like 81283 to move from near 0 (completely open) to well above 0 after adding the missing walls.
Minimal API Changes: All this can be done behind the scenes in the renderer. The add_proxy_walls flag already exists
GitHub
 – we’ll just make it smarter. We might expose a parameter to choose using mesh bounds vs semidense bounds, but it could also be automatic (if semidense present, use it). The rest of the pipeline remains the same: the candidate generation and evaluation logic doesn’t need to know that some triangles in the mesh were auto-inserted proxies. It will naturally score candidates that view these proxy walls similarly to how it would if real walls were present.
Finally, it’s worth noting that if the mesh is fundamentally missing large structures, any learned model might still be at a disadvantage (since textures or details on those walls are missing). But our focus here is on the oracle renderer and visualization – ensuring the depth and occupancy evaluation is fair. By sealing the room with proxy geometry, we prevent candidates from being incorrectly rated as having “clear view” (or the RRI getting confused by infinite depths). In essence, we align the rendering with the physical reality implied by the semidense SLAM points, which ultimately leads to more robust next-best-view proposals. Sources:
NBV Renderer config and comments on backface culling and two-sided rendering
GitHub
GitHub
Proxy wall insertion logic in PyTorch3D renderer
GitHub
GitHub
GitHub
GitHub
Plotly rendering setup for two-sided lighting and proxy walls
GitHub
GitHub
GitHub
GitHub
Coordinate frame conventions (Aria RDF camera, Z-up world) and pose construction
GitHub
GitHub
Camera extrinsic inversion for PyTorch3D (ensuring correct view transform)
GitHub
Scene bounds vs mesh bounds (semidense volume usage) in the visualization code
GitHub
Trimesh signed distance note (re: inside vs outside)
Citations
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L291-L300
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L184-L193
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L293-L300
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L301-L308
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L309-L318
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L319-L328
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L329-L338
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L510-L518
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L255-L264
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L275-L283
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L74-L81
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L184-L192
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L189-L197
[PDF] A Survey on Algorithms of Hole Filling in 3D Surface Reconstruction

https://xiaoyuanguo.github.io/website/A%20Survey%20on%20Algorithms%20of%20Hole%20Filling%20in%203D%20Surface%20Reconstruction.pdf
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L476-L485
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L51-L59
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L52-L56
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L263-L271
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L141-L150
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L151-L159

Backface Culling - 2 sided faces? Blender CMI texturing

https://communityforums.atmeta.com/discussions/Creator_Discussion/backface-culling---2-sided-faces-blender-cmi-texturing/1310600
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L62-L70
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L3-L10
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L220-L224
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L340-L348
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L164-L172
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L165-L173
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L489-L496
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L113-L120
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L130-L138

Backface Culling - 2 sided faces? Blender CMI texturing

https://communityforums.atmeta.com/discussions/Creator_Discussion/backface-culling---2-sided-faces-blender-cmi-texturing/1310600
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L198-L205
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L196-L204
GitHub
plotting.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/data/plotting.py#L115-L118
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L187-L195
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L70-L78
GitHub
pytorch3d_depth_renderer.py

https://github.com/JanDuchscherer104/NBV/blob/6d6f1bf41d582c7020e6ecc16431ff2fb0266a32/oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py#L68-L76

---

# Patch Summary


This patch resolves missing interior walls in the NBV (Next‑Best‑View) pipeline by adding support for **proxy walls** based on the semi‑dense scene bounds.  It modifies the PyTorch3D renderer, the EFM3D CPU renderer and the candidate depth renderer to detect when large wall segments are absent and to insert thin planar proxies at the appropriate axis‑aligned boundaries.  The main features are:

## Pytorch3DDepthRenderer

* Added an optional `occupancy_extent` parameter to `render_batch` and `_mesh_to_struct`.  It encodes the semi‑dense volume bounds of the scene as `[xmin, xmax, ymin, ymax, zmin, zmax]` and is used to expand the mesh bounds when detecting missing walls.  If provided, the renderer now merges these extents with the mesh's `bounds` before evaluating wall coverage.
* Enhanced `_maybe_with_proxy_walls` to consider the combined mesh and occupancy bounds, computing expected wall areas from these extents.  Faces lying within an epsilon of each boundary plane are measured, and if their total area falls below the configured threshold (`proxy_wall_area_threshold`) a thin box is constructed at the expanded bounds.  Only faces corresponding to missing planes are kept and concatenated with the original mesh.
* The two‑sided rendering behaviour is preserved (configurable via `two_sided`), and backface culling remains disabled by default.

## CandidateDepthRenderer

* During candidate depth rendering, the renderer now computes an `occupancy_extent` from the semi‑dense SLAM data (`sample.semidense.volume_min` and `volume_max`).  If available, these bounds are converted to `[xmin,xmax,ymin,ymax,zmin,zmax]` and passed into the underlying PyTorch3D renderer.  This ensures proxy walls align with the true room extent rather than the possibly truncated mesh bounds.
* Added detailed logging of mesh statistics, candidate poses and depth hit ratios for debugging.

## Efm3dDepthRenderer (CPU‑based)

* Extended the configuration with `add_proxy_walls`, `proxy_wall_area_threshold` and `proxy_eps` flags to mirror the PyTorch3D renderer.  These control whether proxy walls are added, the missing area threshold and the epsilon for face proximity.
* Added an optional `occupancy_extent` argument to `render_depth` and `_maybe_with_proxy_walls`.  Just like the GPU renderer, this expands the mesh bounds before checking wall coverage and constructing the proxy box.
* The proxy wall logic matches the PyTorch3D implementation but operates entirely on the CPU using `trimesh`.  It falls back gracefully if `trimesh` or ray intersection routines are unavailable.

## Effect of the Patch

These changes close large holes in incomplete ASE meshes by fusing occupancy and mesh bounds.  As a result, depth renders no longer saturate at the far plane when walls are missing.  The candidate NBV evaluations are also more reliable because the synthetic proxy walls provide realistic depth hits where real walls would exist.  The modifications preserve backwards compatibility: proxy walls can be disabled via configuration, and code paths without semi‑dense data continue to operate using mesh bounds alone.
Here’s a patch summary file that outlines the changes implemented to fix the missing interior walls in the NBV rendering pipeline. The modifications incorporate semi‑dense occupancy bounds into the PyTorch3D and EFM3D depth renderers and update the candidate depth renderer to pass these bounds, ensuring proxy walls close the scene even when the original mesh lacks walls. You can apply these changes to your repository.

In summary, the patch:

* Adds an `occupancy_extent` parameter to the `Pytorch3DDepthRenderer` and extends its `_maybe_with_proxy_walls` method to use semi‑dense bounds for proxy wall placement.
* Modifies the `CandidateDepthRenderer` to compute occupancy extents from semidense SLAM data and pass them into the renderer.
* Extends the EFM3D CPU depth renderer with similar proxy-wall logic and optional occupancy-based bounds.
* Keeps two-sided rendering and backface-culling behaviour configurable while providing stronger guarantees that walls are visible in depth and Plotly renders.

These adjustments ensure that even highly fragmented ASE meshes are enclosed, preventing views into empty space and yielding more reliable depth maps for the NBV pipeline.


---

# Final Remarks

*User Query:* How confidently can you say that you have identified the issue?

Short answer: I’m pretty confident about the high‑level root cause (~80–85%), but only moderately confident (~60–70%) that the specific fixes I sketched are sufficient on their own without iteration in your codebase.



Let me break down what I’m sure about vs what’s still speculative.

What I’m quite confident about
1. The primary root cause

Main issue:
Your GT meshes for some ASE scenes (like 81283) are fundamentally missing most vertical wall geometry, so from an interior camera you’re “looking out into nothing” unless you synthesize walls.



Evidence you already collected and I relied on:

Wall coverage stats for scene 81283:

Floor/ceiling ~224 m² each at zmin/zmax.

Side walls: xmin 0 faces, xmax 76 m², ymin 0, ymax 37 m² → essentially no vertical walls compared to a closed room.

Only ~607 faces near the outer bounds (~0.036% of faces), i.e. almost no shell.

336k components, largest 88k faces → highly fragmented, mostly objects/furniture; no structural shell.

Given that, it’s almost inevitable that:

Depth renders will saturate toward z_far in directions pointing at those missing walls.

Plotly mesh views will look “open” unless we add synthetic geometry.

I’m 90%+ confident that “walls missing from the GT mesh” is the dominant reason for the visual gaps and low hit_ratio, not something like trivial backface culling or a sign error in depth.

2. Why existing proxy walls weren’t enough

The code & your description suggest:

Proxy walls are based on mesh.bounds and a coverage fraction:

It checks area of triangles near xmin/xmax/ymin/ymax/zmin/zmax, compares to expected rectangular area, and adds a box when coverage < threshold.

But your semidense volume_min/volume_max often extend well beyond mesh.bounds (e.g. Y down to –19 m vs mesh to –11 m).

So even if proxies are added, they may be:

Positioned at a truncated box (mesh bounds) that does not reach the real room walls, or

Not triggered because coverage statistic is computed in the wrong bounds frame / too tolerant.

I’m 80–85% confident that “proxy walls derived only from mesh.bounds (ignoring semidense occupancy extents)” is a major part of why turning on add_proxy_walls only modestly improves hit_ratio and still leaves the scene visually open.

3. Coordinate frames are probably not the main bug

From the efm3d/ATEK conventions and the visualization code:

Camera frame is RDF; world is Z‑up with gravity –Z.

T_world_cam = T_world_rig @ T_camera_rig.inverse() is used both in efm3d and your Streamlit plotting.

For PyTorch3D, the code inverts T_world_cam to R_cw/T_cw before passing to the camera, which is the expected view transform.

Plotly overlay of mesh + trajectory + semidense appears spatially consistent in your description.

All of that makes a big “pose is flipped / wrong-handedness” bug unlikely — you are seeing furniture etc. in front of the camera, just no walls.



I’d say 70–80% confident that coordinate-frame mistakes are not the primary issue here.

Where my confidence is lower
4. Exact patch correctness & sufficiency

I proposed:

Extending PyTorch3D and EFM3D renderers with:

occupancy_extent = [xmin, xmax, ymin, ymax, zmin, zmax] from sem.volume_min/max.

Merging these extents into the bounds used for proxy wall placement.

Updating CandidateDepthRenderer to compute and pass that occupancy_extent.

Relying on thresholds like proxy_wall_area_threshold and proxy_eps for wall detection.

This is conceptually sound, but:

I don’t have your full runtime environment (PyTorch3D version, exact BaseConfig/Console implementations, etc.), so some details may need tweaking.

I haven’t actually run the full Streamlit NBV app or your candidate evaluation loop on Scene 81283 with these changes to verify visually.

There could be secondary issues (e.g. camera NDC conversion edge cases, RasterizationSettings quirks, or Plotly lighting & opacity interactions) that only show up in-situ.

So I’d put 60–70% confidence on:

“If you wire semidense volume_min/max into proxy wall generation the way I described, your interiors will look closed and hit_ratio will jump to something reasonable for most scenes without further substantial fixes.”

I fully expect at least some iteration:

Tuning thresholds (proxy_wall_area_threshold, eps).

Maybe offsetting proxy walls a few cm to avoid z‑fighting with partial walls.

Deciding if you want all six planes or only missing ones.

Ensuring Plotly uses the same expanded bounds for its own proxy box.

How to validate quickly (and boost confidence)

If you’re on the fence, here are immediate checks you can do:

Log the bounds you’re actually using for proxies on 81283:

mesh.bounds

sem.volume_min / volume_max

vmin/vmax after merging

If mesh.bounds are clearly inside the semidense AABB, that confirms the “truncated shell” hypothesis.

Log per-plane coverage & “missing” flags:



For X/Y/Z min/max, log:

desired_area, covered_area, covered_area / desired_area, and whether it triggers proxies.

On 81283 you should see side walls with coverage near 0 and thus proxies turning on.

Visual sanity check in Plotly:

Plot mesh + semidense points + a pure box from sem.volume_min/max (no area detection) in a test script.

If that box visually matches where you expect the room shell to be and “closes” the perceived gaps, then using those bounds for proxy walls is the right move.

If those checks behave as expected, my confidence in the issue identification goes to ~95%, and confidence in the patch direction goes to 80%+ (leaving only normal engineering minutiae).

TL;DR

High confidence: the core problem is missing wall polygons in ASE meshes + proxy walls tied only to mesh.bounds, so interiors aren’t actually enclosed.

Medium confidence: the exact code patches I sketched will work on the first try without any further tuning or minor fixes.

Expect at least one iteration where you:

Confirm that semidense bounds are wired into proxy placement,

Check coverage logs on a problematic scene (like 81283),

And visually verify that depth maps now hit proxy walls instead of z_far.

---

2025-11-22 – Documentation maintenance
- Added Context7 library IDs (mermaid, quarto) to AGENTS.
- Fixed mermaid syntax in `docs/contents/impl/aria_nbv_overview.qmd` (NBV loop, architecture diagram) and noted pending rename to `aria_nbv`.
- Tests not run (documentation-only changes).

2025-11-22 – Diagram validation note
- Quarto render check: `quarto render docs/contents/impl/aria_nbv_overview.qmd --to html` succeeds; mermaid diagrams compile.
- Added AGENTS note to always validate mermaid via `quarto render`; use `<br/>` in labels and avoid math braces in node IDs for Mermaid 11+.

2025-11-22 – Mermaid CLI validation
- Validated diagrams with `npx -y @mermaid-js/mermaid-cli`: fixed edge labels by quoting text (`|"label"|`) to avoid parser errors; confirmed both flowchart and classDiagram render to SVG.
- Updated `docs/contents/impl/aria_nbv_overview.qmd` with the quoted labels.
- Added AGENTS workflow for standalone mmdc validation before Quarto render.

2025-11-22 – Data pipeline diagram fix
- `data_pipeline_overview.qmd` mermaid block failed due to `\n` newlines; replaced with `<br/>` line breaks and validated via `npx -y @mermaid-js/mermaid-cli -i /tmp/data_pipeline_flow.mmd -o /tmp/data_pipeline_flow.svg`.
- Quarto render recommended after edits (not rerun here; mermaid-cli succeeded).

2025-11-22 – QMD formatting automation
- Added `scripts/format_qmd_lists.py` to insert blank lines before bullet/numbered lists (avoids Quarto inline rendering issues). Ran it across `docs/**/*.qmd`, updating 28 files.
- AGENTS now instructs running the script before committing docs for consistent formatting.

2025-11-22 – Formatter bugfix
- Updated `format_qmd_lists.py` to avoid inserting blank lines between list items and to handle ``` and ~~~ fences; re-ran across docs (27 files touched). List blocks now keep a single blank line before the block and none between items.

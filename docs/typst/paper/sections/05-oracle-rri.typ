= Oracle RRI Computation

#import "/typst/shared/macros.typ": *

Oracle RRI labels are computed offline by rendering candidate depth maps from
ground-truth meshes and fusing these points with the current reconstruction.
The pipeline is implemented with PyTorch3D rasterization and EFM3D utilities
for unprojection, point fusion, and Chamfer evaluation.

== Candidate depth rendering

For each candidate pose $q$, we render a depth map from the GT mesh using a
metric z-buffer. Let $D_q$ denote the depth map and $C_q$ the camera
intrinsics. We backproject valid depth pixels into world coordinates to obtain
$#sym_points _q$. Missing pixels are masked out and do not contribute to the fused point
cloud. This ensures that the oracle computation respects the same geometric
constraints as the candidate camera.

Depth-render diagnostics and example candidates are provided in the appendix
to keep the main text focused on the pipeline details.

== Point cloud fusion and evaluation

We fuse the candidate point cloud with the current reconstruction,
$#sym_points _(t union q) = #sym_points _t union #sym_points _q$, and evaluate the Chamfer distance to the mesh.
The oracle RRI is computed using the definition above. We also log the intermediate
completeness and accuracy components of the Chamfer distance to diagnose
failure modes (e.g., views that look into empty space). The resulting RRI values
serve as supervised labels for the VIN model.

Implementation note: distances are evaluated with differentiable
point-to-triangle and triangle-to-point primitives (averaged over points and
faces). This yields the same accuracy (P #sym.arrow.r M) and completeness (M
#sym.arrow.r P) split reported by ATEK-style surface reconstruction evaluation.

== Pipeline structure

The oracle label computation consists of four stages:

- sample candidate poses around the current rig pose under collision and
  free-space constraints,
- render candidate depth maps from the ground-truth mesh and mask invalid
  pixels,
- backproject valid depths into world points, fuse with the semi-dense SLAM
  reconstruction, and form the candidate-augmented point set,
- compute Chamfer-based reconstruction quality before/after adding each
  candidate, yielding per-candidate RRI labels.

This separation ensures that rendering and label computation are decoupled from
the VIN model. The VIN architecture consumes only the resulting labels and
backbone features; it never accesses GT meshes or depth renders directly.

The oracle is expensive because it requires rendering and point-to-mesh
distance computation for each candidate. This motivates a lightweight model
that predicts RRI from precomputed scene features without rendering.

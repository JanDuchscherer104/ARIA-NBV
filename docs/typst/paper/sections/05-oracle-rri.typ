= Oracle RRI Computation

#import "../../shared/macros.typ": *

Oracle RRI labels are computed offline by rendering candidate depth maps from
ground-truth meshes and fusing these points with the current reconstruction.
The pipeline is implemented with PyTorch3D rasterization and EFM3D utilities
for unprojection, point fusion, and Chamfer evaluation.

== Candidate depth rendering

For each candidate pose $q$, we render a depth map from the GT mesh using a
metric z-buffer. Let $D_q$ denote the depth map and $C_q$ the camera
intrinsics. We backproject valid depth pixels into world coordinates to obtain
$#symb.oracle.points_q$. Missing pixels are masked out and do not contribute to the fused point
cloud. This ensures that the oracle computation respects the same geometric
constraints as the candidate camera.

Depth-render diagnostics and example candidates are provided in the appendix
to keep the main text focused on the pipeline details.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    image("/figures/app/candidate_renders.png", width: 100%),
    image("/figures/app/rri_forward.png", width: 100%),
  ),
  caption: [Streamlit diagnostics: candidate depth renders (left) and oracle RRI scoring (right); config in @tab:oracle-label-config.],
) <fig:oracle-dashboard>

== Point cloud fusion and evaluation

We fuse the candidate point cloud with the current reconstruction,
$#(symb.oracle.points)_(t union q) = #(symb.oracle.points)_t union #symb.oracle.points_q$, and evaluate the Chamfer distance to the mesh.
The oracle RRI is computed using the definition above. We also log the intermediate
completeness and accuracy components of the Chamfer distance to diagnose
failure modes (e.g., views that look into empty space). The resulting RRI values
serve as supervised labels for future learning-based candidate scoring.

In practice, the per-candidate oracle scores are highly skewed: most candidates
produce only marginal improvements, while a small subset yields large gains
depending on whether the view opens new surface regions or refines previously
observed geometry. For example, in a cached oracle run (scene 81056, sample
000022) the median candidate achieves $0.0116$ RRI while the best candidate
achieves $0.766$ RRI, corresponding to a reduction of bidirectional
point↔mesh error from $2.286$ to $0.535$.

#figure(
  image("/figures/app/rri_hist_81056_000022.png", width: 100%),
  caption: [Oracle RRI distribution across candidates for one example snippet (scene 81056, sample 000022).],
) <fig:rri-hist-example>

We evaluate both directional terms with mean squared point-to-triangle and
triangle-to-point distances (averaged over points and faces). This yields the
same accuracy (P #sym.arrow.r M) and completeness (M #sym.arrow.r P) split
reported by ATEK-style surface reconstruction evaluation.

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

In practice, candidate scoring is batched on GPU: the ground-truth mesh is
cropped to an occupancy-aligned bounding box shared across candidates and the
point↔mesh distances are evaluated for all candidates in a single forward pass
whenever memory permits.

This separation ensures that rendering and label computation are decoupled from
any learned NBV policy. A future VIN-style architecture would consume only the
resulting labels and scene features; it would never access GT meshes or depth
renders directly.

The oracle is expensive because it requires rendering and point-to-mesh
distance computation for each candidate. This motivates a lightweight model
that predicts RRI from precomputed scene features without rendering.

#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

#import "/typst/shared/macros.typ": *

= System Pipeline and Implementation

Our NBV system follows a modular
pipeline from data ingestion to candidate scoring. The key stages are
summarized in @tab:pipeline and visualized in @fig:candidate-poses.


#figure(
  image("/figures/candidate_pose.png", width: 100%),
  caption: [Candidate pose sampling around the reference pose (diagnostic view).],
) <fig:candidate-poses>

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Pipeline modules and data flow.],
  table(
    columns: (14em, auto, auto),
    align: (left, left, left),
    toprule(),
    table.header(
      [
        Module
      ],
      [Inputs],
      [Outputs],
    ),
    midrule(), [Dataset loader], [ASE shards, mesh],
    [Snippet window (streams, poses, semidense points, mesh)], [Candidate generation], [Rig pose, mesh, bounds],
    [Candidate poses], [Depth rendering], [Candidates, mesh, cameras],
    [Depth maps, masks], [Oracle RRI], [$#sym_points _t, #sym_points _q, #sym_mesh$],
    [RRI labels], [VIN inference], [EVL features, poses],
    [Ordinal RRI scores], bottomrule(),
  ),
) <tab:pipeline>

== Candidate generation

Candidates are sampled on a constrained spherical shell around the current rig
pose, with configurable radius and elevation limits. We enforce collision
constraints using free-space and mesh intersection checks. The candidate set is
kept discrete for training stability and reproducibility, following the
VIN-NBV strategy.

Trajectory and sampling diagnostics are summarized in the appendix.

== Rendering and backprojection

We render candidate depth maps with PyTorch3D and backproject depths to world
points using EFM3D camera utilities. These points are fused with the
semi-dense SLAM reconstruction to form the candidate-augmented point cloud. The
rendering step is expensive but performed offline to generate oracle labels.

We explicitly track valid depth pixels with a per-pixel validity mask, ensuring
that miss pixels or z-buffer failures do not pollute the candidate point cloud.
This mask is also useful for debugging candidate views that appear to look
through walls or miss the mesh entirely.

== Streamlit diagnostics

A Streamlit dashboard exposes the pipeline stages with cached intermediate
results. We inspect candidate distributions, depth render quality, and VIN
intermediate features to identify failure modes and verify coordinate
conventions.

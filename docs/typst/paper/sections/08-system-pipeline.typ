#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

#import "/typst/shared/macros.typ": *

= System Pipeline and Implementation

Our pipeline follows a modular
path from data ingestion to oracle label computation (and future candidate
scoring). The key stages are
summarized in @tab:pipeline and visualized in @fig:candidate-poses.


#figure(
  image("/figures/app/cand_frusta_kappa4_r06-29.png", width: 100%),
  caption: [Candidate frusta sampled around a reference pose (Streamlit diagnostics view; config in @tab:oracle-label-config).],
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
    [Snippet window (streams, poses, semi-dense points, mesh)], [Candidate generation], [Rig pose, mesh, bounds],
    [Candidate poses], [Depth rendering], [Candidates, mesh, cameras],
    [Depth maps, masks], [Oracle RRI], [$#sym_points _t, #sym_points _q, #sym_mesh$],
    [RRI labels (continuous + ordinal)], [Future: learned scorer], [oracle labels, scene features],
    [Predicted ordinal scores], bottomrule(),
  ),
) <tab:pipeline>

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Representative oracle label configuration used in the Streamlit figures.],
  table(
    columns: (18em, auto),
    align: (left, left),
    toprule(),
    table.header([Parameter], [Value]),
    midrule(),
    [Candidate count (requested)], [32],
    [Candidate shell radius], [[0.6, 2.9] m],
    [Elevation range], [-15° to 25°],
    [Azimuth spread], [170°],
    [Direction sampler], [uniform sphere ($kappa = 4$ when biased)],
    [Min distance to mesh], [$0.4$ m],
    [Collision + free-space checks], [enabled],
    [View direction mode], [radial away],
    [View jitter caps], [60° az / 30° elev],
    [View roll jitter], [0°],
    [Depth renderer (max kept)], [16],
    [Depth z-range], [znear=1e-3, zfar=20],
    [Oracle reduction], [mean over triangles (cropped AABB)],
    bottomrule(),
  ),
) <tab:oracle-label-config>

== Candidate generation

Candidates are sampled on a constrained spherical shell around the current rig
pose, with configurable radius and elevation limits. We enforce collision
constraints using free-space and mesh intersection checks. The candidate set is
kept discrete to enable reproducible oracle label computation and to match the
candidate-ranking formulation of VIN-NBV @VIN-NBV-frahm2025.

Trajectory and sampling diagnostics are summarized in the appendix.

== Rendering and backprojection

We render candidate depth maps with PyTorch3D and backproject depths to world
points using EFM3D camera utilities. These points are fused with the
semi-dense SLAM reconstruction to form the candidate-augmented point cloud. The
rendering step is expensive but performed offline to generate oracle labels.

We explicitly track valid depth pixels with a per-pixel validity mask, ensuring
that missing pixels or z-buffer failures do not pollute the candidate point cloud.
This mask is also useful for debugging candidate views that appear to look
through walls or miss the mesh entirely.

== Streamlit diagnostics

A Streamlit dashboard exposes the pipeline stages with cached intermediate
results. We inspect candidate distributions, depth render quality, and oracle
RRI behavior to identify failure modes and verify coordinate conventions.

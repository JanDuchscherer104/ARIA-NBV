= Appendix: Candidate Sampling Gallery

This appendix collects representative diagnostic figures from the candidate
sampling and rendering pipeline. The gallery is intended for qualitative
validation of pose distributions, collision checks, and view frusta.

#figure(
  image("/figures/app/traj.png", width: 100%),
  caption: [Trajectory overview from the diagnostics dashboard.]
) <fig:traj>

#figure(
  image("/figures/app/candidate_renders.png", width: 100%),
  caption: [Example candidate depth renders from the GT mesh.]
) <fig:candidate-renders>

#figure(
  image("/figures/app/rri_forward.png", width: 100%),
  caption: [RRI forward diagnostics and per-candidate scoring.]
) <fig:rri-forward>

#figure(
  image("/figures/app/render_frusta.png", width: 100%),
  caption: [Rendered frusta with depth overlay for candidate validation.]
) <fig:render-frusta>

#figure(
  image("/figures/app/semidense.png", width: 100%),
  caption: [Semidense point cloud overlay on RGB for sanity checking.]
) <fig:semidense-overlay>

#figure(
  image("/figures/ase_semi_dense.png", width: 100%),
  caption: [ASE semi-dense point cloud example from a single snippet.],
) <fig:ase-semidense>

#figure(
  image("/figures/efm3d/evl_output_summary.png", width: 100%),
  caption: [EVL output summary used to form the scene field.],
) <fig:evl-summary>

#figure(
  image("/figures/app/depth_hist.png", width: 100%),
  caption: [Depth histogram diagnostics for rendered candidates.]
) <fig:depth-hist>

#figure(
  image("/figures/app/first_final_frames.png", width: 100%),
  caption: [Early vs. late frames in ASE snippets highlighting stage effects.]
) <fig:stage-frames>

#figure(
  image("/figures/app/candidate_frusta_look_away_tilt.png", width: 100%),
  caption: [Candidate frusta visualization with roll and elevation variation.]
) <fig:frusta>

#figure(
  image("/figures/app/depth_render.png", width: 100%),
  caption: [Depth render and frustum alignment for a sampled candidate.]
) <fig:depth-render>

#figure(
  image("/figures/app/gt_obbs.png", width: 100%),
  caption: [Ground-truth OBBs from ASE snippets (entity-aware context).]
) <fig:gt-obbs>

#figure(
  image("/figures/app/cand_away.png", width: 100%),
  caption: [Candidate set oriented away from the reference trajectory.]
) <fig:cand-away>

#figure(
  image("/figures/app/frusta_away.png", width: 100%),
  caption: [Candidate frusta pointing away from the reference pose.]
) <fig:frusta-away>

#figure(
  image("/figures/app/dir_dist_full_circ.png", width: 100%),
  caption: [Direction and distance coverage over the full circle.]
) <fig:dir-dist>

#figure(
  image("/figures/app/candidates_full_circ.png", width: 100%),
  caption: [Candidate distribution over full azimuth range.]
) <fig:cand-full>

#figure(
  image("/figures/app/collision.png", width: 100%),
  caption: [Collision checking diagnostics with mesh intersections.]
) <fig:collision>

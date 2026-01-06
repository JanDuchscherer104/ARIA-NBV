// Oracle RRI pipeline slides (ASE -> candidates -> rendering -> backprojection -> scoring)
// Style baseline: docs/typst/slides/slides_2.typ

#import "@preview/definitely-not-isec-slides:1.0.1": *
#import "@preview/muchpdf:0.1.1": muchpdf

// Import shared macros and symbols
#import "../shared/macros.typ": *

#let fig_path = "../../figures/"

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: false,
  institute: [HM],
  logo: [#image(fig_path + "hm-logo.svg", width: 2cm)],
  config-info(
    title: [Oracle RRI Pipeline in ASE],
    subtitle: [Data #sym.arrow.r Candidates #sym.arrow.r Rendering #sym.arrow.r Backprojection #sym.arrow.r Metrics],
    authors: [*Jan Duchscherer*],
    extra: [VCML Seminar WS24/25],
    footer: [
      #grid(
        columns: (1fr, auto, 1fr),
        align: bottom,
        align(left)[Jan Duchscherer],
        align(center)[VCML Seminar WS24/25],
        align(right)[#datetime.today().display("[day padding:none]. [month repr:short] [year]")],
      )
    ],
    download-qr: "",
  ),
  config-common(handout: false),
  config-colors(primary: rgb("fc5555")),
)

// Global text size
#set text(size: 17pt)

// Style links to be blue and underlined
#show link: set text(fill: blue)
#show link: it => underline(it)

// ---------------------------------------------------------------------------
// Title
// ---------------------------------------------------------------------------

#title-slide()

// ---------------------------------------------------------------------------
// Pipeline overview
// ---------------------------------------------------------------------------

#section-slide(
  title: [Oracle RRI Pipeline],
  subtitle: [Reusable compute path for dashboard, preprocessing, and training #image(fig_path + "impl/label_batch.png", width: 60%)],
)

#slide(title: [Oracle RRI Pipeline])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Inputs (per snippet)])[
        - `EfmSnippetView`: cameras, trajectory, semi-dense points, optional GT/OBBs
        - GT mesh tensors: `mesh_verts` / `mesh_faces`
        - Reference pose: last trajectory pose (or chosen frame)
      ]
      #color-block(title: [Orchestrator])[
        - `OracleRriLabeler` composes three stage configs:
          + `CandidateViewGeneratorConfig` (pose sampling + pruning)
          + `CandidateDepthRendererConfig` (depth simulation from GT mesh)
          + `OracleRRIConfig` (point #sym.arrow.l.r mesh distances #sym.arrow.r #RRI)
        - Pipeline knob: `backprojection_stride` (pixel subsampling)
      ]
    ],
    [
      #color-block(title: [Stages])[
        1. *Candidate Generation*: sample + orient + prune `PoseTW["C 12"]`
        2. *Depth Rendering*: render `depths["C H W"]` (+ valid mask)
        3. *Backprojection*: depth hits #sym.arrow.r `CandidatePointClouds`
        4. *Oracle Scoring*: `RriResult` per candidate
      ]
      #color-block(title: [Outputs])[
        - `OracleRriLabelBatch` contains:
          + `candidates` (`CandidateSamplingResult`)
          + `depths` (`CandidateDepths`)
          + `candidate_pcs` (`CandidatePointClouds`)
          + `rri` (`RriResult`)
      ]
    ],
  )

  #quote-block(color: rgb("#ff0000"))[
    Currently engineered for plotting and debugging in the streamlit app. Results contain too many intermediate tensors for optimal batching per snippet.
  ]
]

// ---------------------------------------------------------------------------
// Data handling (download -> dataset -> typed views -> mesh processing)
// ---------------------------------------------------------------------------

// #section-slide(
//   title: [Data Handling],
//   subtitle: [Downloading ASE + streaming ATEK -> typed `EfmSnippetView`],
// )[
//   #figure(image(fig_path + "scene-script/ase_modalities.jpg", width: 80%), caption: [@SceneScript-avetisyan2024])
// ]

// #slide(title: [ASE Modalities Used by oracle_rri])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1.2cm,
//     [
//       #color-block(title: [Core modalities])[
//         - *Trajectory*: `PoseTW` world$<-$rig for each timestamp
//         - *Semi-dense SLAM points*: sparse but informative geometry cues
//         - *GT mesh*: watertight-ish surface geometry (oracle target)
//         - *Camera calibration*: `CameraTW` intrinsics/extrinsics per stream
//       ]
//       #color-block(title: [Why we need all of them])[
//         - Candidate poses must be sampled in a *physically consistent* world frame.
//         - Rendering needs a mesh + camera model.
//         - Backprojection needs rendered depth + the exact camera used to render.
//         - RRI uses point<->mesh distances, so both point sets and mesh must share coordinates.
//       ]
//     ],
//     [
//       #color-block(title: [Typed views])[
//         - `EfmCameraView`: images + `CameraTW` + timestamps
//         - `EfmTrajectoryView`: `t_world_rig` + helpers (`final_pose()`)
//         - `EfmPointsView`: `points_world`, `dist_std`, bounds, `lengths`
//         - `EfmGTView` / `EfmObbView`: GT timestamps and optional entity boxes
//       ]
//       #quote-block[
//         The dataset yields *views* over an EFM dictionary (zero-copy): move tensors with `.to(device)` without cloning.
//       ]
//     ],
//   )
// ]

// #slide(title: [Unified ASE Downloader CLI])[
//   #color-block(title: [What it does])[
//     - One entrypoint downloads GT meshes + ATEK shards with consistent filtering.
//     - Mesh download: SHA1 validation + ZIP extraction to `.ply`.
//     - ATEK download: write filtered JSON for `download_atek_wds_sequences`.
//   ]
//   #color-block(title: [Modes])[
//     - `download`: pull meshes and/or ATEK for top-N scenes (cap snippets, prefer richest scenes first).
//     - `list`: show scenes with GT meshes and snippet counts (sorted by snippet density).
//   ]
//   #color-block(title: [Usage])[
//     #text(size: 13pt)[
//       - `python -m oracle_rri.data.downloader list --n=8`
//       - `python -m oracle_rri.data.downloader download --n_scenes=5 --max_snippets=2 --c=efm_eval`
//       - Flags: `--skip_meshes`, `--skip_atek`, `--prefer_scenes_with_max_snippets`, `--output-dir`.
//     ]
//   ]
// ]

// #slide(title: [AseEfmDataset + Mesh Handling])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1cm,
//     [
//       #color-block(title: [Dataset wrapper])[
//         - `AseEfmDataset` wraps `load_atek_wds_dataset_as_efm` -> yields `EfmSnippetView`.
//         - Resolves tar shards from `PathConfig`, infers scene/snippet ids.
//         - Optional mesh loading + simplification + caching (`mesh_cache.py`).
//       ]
//       #color-block(title: [Mesh processing])[
//         - Load `.ply` via `trimesh` (vertices + faces).
//         - Optional quadric simplification for speed.
//         - Optional cropping to occupancy bounds for stability/efficiency.
//       ]
//     ],
//     [
//       #color-block(title: [Typed access in practice])[
//         ```python
//         from oracle_rri.data import AseEfmDatasetConfig

//         cfg = AseEfmDatasetConfig(
//             scene_ids=["81048"],
//             atek_variant="efm_eval",
//             load_meshes=True,
//         )
//         ds = cfg.setup_target()
//         sample = next(iter(ds))

//         rgb = sample.camera_rgb
//         traj = sample.trajectory
//         sem = sample.semidense
//         print(sample.has_mesh, sample.mesh is not None)
//         ```
//       ]
//     ],
//   )
// ]

// ---------------------------------------------------------------------------
// Candidate generation (SE(3) poses around a reference pose)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Candidate Generation],
  subtitle: [Generate and prune candidate `PoseTW` around a reference pose],
)[
  #image(fig_path + "app/cand_frusta_kappa4_r06-29.png", width: 100%)
]

#slide(title: [CandidateViewGenerator: Overview])[
  #color-block(title: [Inputs])[
    - Reference pose: last rig/camera pose (optionally gravity-aligned).
    - Occupancy extent (AABB) from semidense bounds or mesh bounds.
    - GT mesh (for clearance + collision pruning).
  ]
  #color-block(title: [Outputs])[
    - Candidate poses: `PoseTW["C 12"]` (world $<-$ cam)
    - Diagnostics: per-rule masks + optional debug stats
    - Stable indexing: map back to global candidate indices for UI alignment
  ]
]

#slide(title: [Candidate Centers (Theory)])[
  #text(size: 15pt)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      [
        #color-block(title: [Sampling distribution])[
          Sample candidate centers on a spherical band in a reference frame:
          $
                r & ~ cal(U)(r_"min", r_"max") \
              phi & ~ cal(U)(-Delta_phi/2, Delta_phi/2) \
                u & ~ cal(U)(sin theta_"min", sin theta_"max") \
            theta & = arcsin(u).
          $

          Unit direction and center:
          $
            bold(s)_i & = (cos theta_i cos phi_i, sin theta_i, cos theta_i sin phi_i) \
            bold(p)_i & = bold(t)_"ref" + bold(R)_"ref" (r_i bold(s)_i) \
            bold(t)_i & = bold(p)_i.
          $
          The orientation is assigned in the *view-direction* stage.
        ]
      ],
      [
        #color-block(title: [Practical options in code])[
          - *Uniform*: area-uniform band sampling (`UNIFORM_SPHERE`).
          - *Forward-biased*: PowerSpherical / von-Mises-Fisher concentration (`kappa`) around forward axis.
          - *Gravity alignment*: remove pitch/roll in the sampling frame (`align_to_gravity=True`).

          #v(0.4em)
          *Intuition*
          - Explore around the last pose while respecting the rig's reachable neighborhood.
          - Limit elevation to remain "human-like" (no ceiling cams).
          - Keep a controllable azimuth spread for forward exploration vs. full sweep.
        ]
      ],
    )
  ]
]

#slide(title: [Candidate Orientations (Theory)])[
  #text(size: 15pt)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      [
        #color-block(title: [Goal])[
          Build a camera rotation `R_world_cam` so each candidate *looks at* a target (often the last pose),
          while keeping roll stable and respecting the Aria LUF convention:
          - +x: left, +y: up, +z: forward
        ]
        #color-block(title: [Look-at frame])[
          For camera center `p` and target $bold(p)_c$:
          $
            bold(z) & = (bold(p)_c - bold(p)) / (||bold(p)_c - bold(p)||) \
            bold(y) & = bold(u)_"world" \
            bold(x) & = (bold(y) times bold(z)) / (||bold(y) times bold(z)||).
          $
          Then re-orthogonalize: $bold(y) = bold(z) times bold(x)$.
        ]
      ],
      [
        #color-block(title: [What defines the *base* view?])[
          - `view_direction_mode` picks a deterministic base orientation:
            + `radial_away` / `radial_towards`: look along (candidate #sym.arrow.l.r reference) with roll-free world-up.
            + `forward_rig`: reuse the rig rotation.
            + `target_point`: look-at a fixed point.
          - This yields $bold(R)_"base"$.
        ]
        #quote-block(color: rgb("#285f82"))[
          *Roll-free?* One less DOF to worry about!
        ]
      ],
    )
  ]
]

#slide(title: [View-Direction Randomization])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [View Direction Randomization])[
        - Jitter is then applied as a *right-multiplicative* delta:
          $
            bold(R)_("final") = bold(R)_("base") dot bold(R)_("delta").
          $
      ]
      #quote-block(color: rgb("#285f82"))[
        This makes jitter \"relative to the candidate's current view direction\": `0deg` means \"keep the base view\".
      ]
    ],
    [
      #color-block(title: [Caps vs sampler (priority order)])[
        *bounded uniform* jitter around the base direction via `view_max_azimuth_deg`,  `view_max_elevation_deg` and `view_roll_jitter_deg`:

        #v(0.2em)
        Sample angles (caps in radians):
        $
          bullet & ~ cal(U)(-bullet_"max" / 2 , +bullet_"max" / 2 ), bullet in {psi, theta, phi}
        $

        Convert to a valid SO(3) rotation and compose:

        Build a right-multiplicative delta rotation and compose:
        $
          bold(R)_"delta" = bold(R)_z(psi) bold(R)_y(theta) bold(R)_x(phi) \
        $

        // Optional roll jitter:
        // $
        //   gamma ~ cal(U)(-gamma_"max", +gamma_"max") \
        //   bold(R)_"delta" <- bold(R)_"delta" bold(R)_z(gamma).
        // $

        // #v(0.2em)
        // (If both caps are 0, `view_sampling_strategy` can instead sample $bold(z)_"delta"$ on $S^2$.)
      ]
    ],
  )
]



#slide(title: [Pose-aligned vs Gravity-aligned Sampling])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #color-block(title: [Pose-aligned (`align_to_gravity=false`)])[
        #image(fig_path + "app/view_dirs_pose_align.png", width: 100%)
      ]
    ],
    [
      #color-block(title: [Gravity-aligned (`align_to_gravity=true`)])[
        #image(fig_path + "app/view_dirs_gravity_align.png", width: 100%)
      ]
    ],
  )
]

#slide(title: [Pose-aligned vs Gravity-aligned Sampling])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #color-block(title: [Pose-aligned (`align_to_gravity=false`)])[
        #image(fig_path + "app/cand_positions_pose_align.png", width: 100%)
      ]
    ],
    [
      #color-block(title: [Gravity-aligned (`align_to_gravity=true`)])[
        #image(fig_path + "app/cand_positions_gravity_align.png", width: 100%)
      ]
    ],
  )
]

#slide(title: [Roll Jitter: What Changes?])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #color-block(title: [No roll jitter (`view_roll_jitter_deg=0`)])[
        #image(
          fig_path + "app/dir_dists_full_az_elmin_neg15_elmax_25_unfi_sphere_radial_away_no_jitter_fixed.png",
          width: 100%,
        )
      ]
    ],
    [
      #color-block(title: [With roll jitter (`view_roll_jitter_deg~=17deg`)])[
        #image(
          fig_path + "app/dir_dists_full_az_elmin_neg15_elmax_25_unfi_sphere_radial_away_17_roll_jitter.png",
          width: 100%,
        )
      ]
    ],
  )
  #quote-block(color: rgb("#285f82"))[
    Roll jitter changes the *twist* about the forward axis (roll histogram spreads), while azimuth/elevation of view
    directions should not be affected.
  ]
]

#slide(title: [Forward Bias: Candidate Center Distributions])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #color-block(title: [Forward-biased (PowerSpherical, `kappa=4`, `r=0.6..2.9`)])[
        #image(fig_path + "app/cand_pose_kappa4_r06-29.png", width: 100%)
      ]
    ],
    [
      #color-block(title: [Narrower azimuth + stronger pruning (`delta_az~=165deg`, `min_dist~=1m`)])[
        #image(fig_path + "app/cand_pose_az165_mindist1_r06-2.4.png", width: 100%)
      ]
    ],
  )
  #quote-block(color: rgb("#285f82"))[
    Forward bias (PowerSpherical, `kappa>0`) or (UniformSpherical, with azimuth restriction) concentrates candidate centers near the current forward direction.
    This matches realistic motion (users rarely turn 180#sym.degree for the next view), without forward bias, candiates looking in the opposite direciton of the reference will dominate RRI.
  ]
]

#slide(title: [Forward-biased Candidates: Frusta in Scene])[
  #grid(
    columns: (1.25fr, 0.75fr),
    gutter: 0.8cm,
    [
      #image(fig_path + "app/cand_frusta_kappa4_r06-29.png", width: 100%)
    ],
    [
      #color-block(title: [Reading the plot])[
        - Blue points: candidate centers after pruning.
        - Red wireframes: rendered camera frusta (subset).
        - Axes: reference frame at the sampling origin.
        - In `radial_away`, the frusta optical axes align with the center direction (up to optional view jitter).
      ]
    ],
  )
]

#slide(title: [Rule: Free-Space AABB])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Criterion])[
        Constrain candidate centers to an occupancy extent:
        $
          x_"min" <= x_i <= x_"max", \
          y_"min" <= y_i <= y_"max", \
          z_"min" <= z_i <= z_"max".
        $
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Prevent candidates from leaking outside the local room/scene bounds.
        - Bounds come from semidense metadata (`volume_min/max`) or mesh crop bounds.
        - Cheap to evaluate (first pruning stage).
      ]
    ],
  )
]


#slide(title: [Rule: Min Distance to Mesh])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Criterion])[
        For candidate center $bold(c)_i$ and mesh $bold(cal(M))$:
        $
          d_i = min_(bold(x) in bold(cal(M))) ||bold(c)_i - bold(x)||_2.
        $
        Keep iff $d_i > d_"min"$.
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Reject viewpoints that start inside/too close to GT geometry.
        - Prevents unstable renders (near-plane clipping explosions).
        - Implemented via PyTorch3D point #sym.arrow.l.r mesh distance or trimesh proximity queries.
      ]
    ],
  )
]

#slide(title: [Rule: Path Collision])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Criterion])[
        Straight-line segment from reference pose $bold(o)$ to candidate center $bold(p)_i$:
        $
          bold(r)_(i)(t) = bold(o) + t hat(bold(d)_i),
          t in [0, ||bold(p)_i - bold(o)||].
        $
        Reject if the segment intersects the mesh (optionally with clearance).
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Enforce collision-free translation in a simple
        - Backends:
          + PyTorch3D distance sampling along the segment (GPU)
          + Trimesh / PyEmbree ray intersector (CPU)
      ]
    ],
  )
]



// ---------------------------------------------------------------------------
// Depth rendering (simulate candidate observation)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Depth Rendering],
  subtitle: [Render candidate depth maps from the GT mesh

    #image(fig_path + "app/depth_render.png", width: 60%)],
)

#slide(title: [CandidateDepthRenderer: Responsibilities])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Candidate selection])[
        - Render only a subset for performance:
          + oversample `oversample_factor`
          + filter out invalid renders
          + cap to `max_candidates_final`
      ]
      #color-block(title: [Outputs])[
        - `depths["C H W"]` in metres (metric z-depth)
        - `depths_valid_mask["C H W"]` (hit + near/far clipping)
        - `poses` (world $<-$ cam) and `p3d_cameras` for later backprojection
      ]
    ],
    [
      #color-block(title: [Valid depth mask])[
        We treat a pixel as valid iff:
        - `pix_to_face >= 0`
        - `z > z_near`
        - `z < z_far`
        Misses often appear as `pix_to_face=-1` (and may show `z=-1` in histograms).
      ]

    ],
  )
]

// #slide(title: [PyTorch3D Rasterization (Conceptual)])[
//   #grid(
//     columns: (1fr, 1fr),
//     gutter: 1cm,
//     [
//       #color-block(title: [Camera model])[
//         We use PyTorch3D `PerspectiveCameras` in *screen space* (`in_ndc=false`):
//         $
//           u = f_x X/Z + c_x,
//           v = f_y Y/Z + c_y.
//         $
//         The renderer internally converts screen coordinates to NDC using
//         $s = min(H, W)$ (important for non-square images).
//       ]
//       #color-block(title: [Rasterization output])[
//         `MeshRasterizer` returns fragments:
//         - `pix_to_face`: face index per pixel (hit/miss)
//         - `zbuf`: per-pixel z-depth (metres when configured consistently)
//         - optional barycentric coordinates (not used for oracle depth)
//       ]
//     ],
//     [
//       #color-block(title: [Coordinate conventions])[
//         - PyTorch3D NDC: +X left, +Y up, right-handed.
//         - We keep all core geometry in the physical Aria LUF world frame.
//         - Extrinsics must be consistent with `PoseTW` conventions when building `PerspectiveCameras(R,T)`.
//       ]
//       #color-block(title: [Why rasterization?])[
//         - GPU batch rendering for many candidates
//         - Deterministic depth (z-buffer)
//         - Works directly with mesh tensors (`verts`, `faces`)
//       ]
//     ],
//   )
// ]

// ---------------------------------------------------------------------------
// Backprojection (depth hits -> world-frame point clouds)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Backprojection],
  subtitle: [Depth hits #sym.arrow.r candidate point clouds (world frame)],
)

#slide(title: [Depth #sym.arrow.r 3D Points (Math)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Pixel #sym.arrow.r NDC mapping (PyTorch3D)])[
        For pixel indices `(x,y)` and pixel centers:
        $
          u = x + 0.5,\ v = y + 0.5,\ s = min(H, W).
        $
        Convert to NDC:
        $
          x_"ndc" = -(u - W/2) (2/s),
          y_"ndc" = -(v - H/2) (2/s).
        $
        Form `xy_depth = (x_ndc, y_ndc, z)`.
      ]
    ],
    [
      #color-block(title: [Unprojection])[
        PyTorch3D provides:
        - `unproject_points(xy_depth, world_coordinates=true, from_ndc=true)`
        - Output: `p_world` for each valid pixel.
      ]
    ],
  )
]

#slide(title: [Vectorized Backprojection (Implementation)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Batch strategy])[
        - Rendered depths: `depths["B H W"]`, mask: `mask_valid["B H W"]`.
        - Sample strided pixel grid (stride = `backprojection_stride`).
        - Build batched `xy_depth["B P 3"]` and call `cameras.unproject_points(...)` once.
      ]
      #color-block(title: [Packing valid hits])[
        - Compute `mask["B P"]` for valid pixels.
        - Compact points per batch into:
          + `points["B Pmax 3"]` (padded with NaNs)
          + `lengths["B"]` (valid point counts)
      ]
    ],
    [
      #color-block(title: [Fusion for scoring])[
        - Collapse semi-dense SLAM points:
          + `semidense_points["K 3"]` (NaNs removed, time collapsed)
        - Compute combined bounds:
          + `occupancy_bounds[6]` = union of snippet AABB + candidate PCs + semidense
      ]
      #quote-block[
        Stride is a key compute knob: smaller stride = denser candidate PCs, but more GPU work in both unprojection and point #sym.arrow.l.r mesh distances.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Oracle RRI (point<->mesh distances + normalization)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Oracle RRI Metric],
  subtitle: [Point #sym.arrow.l.r mesh distances (accuracy + completeness) #sym.arrow.r normalized improvement],
)[
  #text(size: 16pt)[Implemented in `oracle_rri/oracle_rri/rri_metrics/`.]
]

#slide(title: [RRI Definition])[
  #color-block(title: [Relative Reconstruction Improvement])[
    For candidate view $bold(q)$:
    $
      "RRI"(bold(q)) =
      (d(bold(cal(P))_t, bold(cal(M))_"GT") - d(bold(cal(P))_{t union bold(q)}, bold(cal(M))_"GT"))
      / d(bold(cal(P))_t, bold(cal(M))_"GT").
    $
    - $bold(cal(P))_t$: semi-dense SLAM reconstruction up to time $t$
    - $bold(cal(P))_{t union bold(q)}$: merged reconstruction after backprojecting candidate depth hits
    - $d(.,.)$: *bidirectional* point #sym.arrow.l.r mesh distance
  ]

  #quote-block(color: rgb("#285f82"))[
    The ratio cancels units and makes scores comparable across scenes and scales.
  ]
]

#slide(title: [Point #sym.arrow.l.r Mesh Distance: Accuracy + Completeness])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Accuracy (P #sym.arrow.r M)])[
        Average distance from reconstruction points to the GT surface:
        $
          d_{bold(cal(P)) -> bold(cal(M))} =
          1/(|bold(cal(P))|) sum_{bold(p) in bold(cal(P))} min_{bold(x) in bold(cal(M))} ||bold(p) - bold(x)||_2^2.
        $
        - Detects *over-reconstruction* (spurious geometry).
        - Implemented via PyTorch3D `point_face_distance` (point #sym.arrow.r triangle).
      ]
    ],
    [
      #color-block(title: [Completeness (M #sym.arrow.r P)])[
        Average distance from GT surface elements to the reconstruction:
        $
          d_{bold(cal(M)) -> bold(cal(P))} approx 1/(|bold(cal(F))|) sum_{bold(f) in bold(cal(F))} min_{bold(p) in bold(cal(P))} d(bold(f), bold(p))^2.
        $
        - Detects *under-reconstruction* (missing geometry).
        - Implemented via PyTorch3D `face_point_distance` (triangle #sym.arrow.r point).
      ]
    ],
  )
  #color-block(title: [Bidirectional])[
    $
      d_{bold(cal(P)) <-> bold(cal(M))} = d_{bold(cal(P)) -> bold(cal(M))} + d_{bold(cal(M)) -> bold(cal(P))}.
    $
  ]
]

#slide(title: [chamfer_point_mesh_batched: Fully Vectorized Breakdown])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Inputs])[
        - Candidate batch: `points["C P 3"]` (padded) + `lengths["C"]`
        - Mesh: `gt_verts["V 3"]`, `gt_faces["F 3"]`
        - Output: `DistanceBreakdown` with `accuracy`, `completeness`, `bidirectional` (per candidate)
      ]
      #color-block(title: [1) Pack points])[
        Turn padded `(C,P,3)` into a packed `(P_tot,3)` representation + index maps:
        ```python
        mask = arange(P)[None, :] < lengths[:, None]    # (C,P) bool
        points_packed = points[mask]                    # (P_tot,3)

        points_first_idx = cumsum(lengths) - lengths    # (C,)
        point_to_cloud_idx = repeat_interleave(arange(C), lengths)  # (P_tot,)
        ```
      ]
    ],
    [
      #color-block(title: [2) Compute distances])[
        PyTorch3D returns packed distances:
        - point #sym.arrow.r face: `(P_tot,)` (each point to nearest triangle)
        - face #sym.arrow.r point: `(F_tot,)` (each face to nearest point)
      ]
      #color-block(title: [3) Reduce per candidate])[
        ```python
        accuracy[c]     = sum(point_to_face[p in c]) / |P|
        completeness[c]  = sum(face_to_point[f in c]) / |F|
        bidirectional[c] = accuracy[c] + completeness[c]
        ```
      ]
    ],
  )
]

#slide(title: [OracleRRI.score: Candidate Batch Computation])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Before / After point sets])[
        - $bold(cal(P))_t$: collapsed semi-dense SLAM points
        - For each candidate view $bold(q)$:
          + $bold(cal(P))_q$: backprojected depth-hit points
          + $bold(cal(P))_{t union bold(q)} = bold(cal(P))_t union bold(cal(P))_q$
      ]
      #color-block(title: [Implementation sketch])[
        1. `dist_before = chamfer_point_mesh(P_t, M_gt)`
        2. Tile `P_t` across candidates and concatenate:
          `P_tq = cat([P_t, P_q])`
        3. `dist_after = chamfer_point_mesh_batched(P_tq, lengths_tq, M_gt)`
        4. `rri = (dist_before - dist_after) / dist_before`
      ]
    ],
    [
      #color-block(title: [Notes & knobs])[
        - *Density matching*: optional voxel downsampling (`voxel_size_m`) to equalize $bold(cal(P))_t$ vs. $bold(cal(P))_{t union bold(q)}$ densities.
        - *Cropping*: mesh can be cropped to `occupancy_bounds` to avoid unrelated geometry.
        - *Numerical safety*: clamp denominator to avoid divide-by-zero.
      ]
      #quote-block(color: rgb("#fc5555"))[
        The oracle is only used to generate training labels; at test time, the learned predictor replaces this expensive pipeline.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// VIN (View Introspection Network): learn to predict RRI from EVL + pose
// ---------------------------------------------------------------------------

#section-slide(
  title: [VIN: View Introspection Network],
  subtitle: [Predict per-candidate #RRI from frozen #EVL features + a shell-aware pose descriptor],
)[
  #text(size: 16pt)[
    Goal: replace the expensive oracle pipeline with a lightweight predictor trained on oracle labels
    (VIN-NBV style ordinal regression @VIN-NBV-frahm2025).
  ]
]

#slide(title: [VIN: I/O + Shapes])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Inputs (symbols)])[
        - Snippet batch size: $B$.
        - Candidate count per snippet: $N_c$.
        - EVL input: raw EFM snippet dict `efm: dict[str, Any]`.
        - Candidate poses: `PoseTW["B N_c 12"]` (world$<-$camera).
        - Reference pose: `PoseTW["B 12"]` (world$<-$rig).
        - Training-aligned candidates: `T_camera_rig` (camera$<-$rig) from `OracleRriLabelBatch.depths.camera`.
      ]
      #color-block(title: [Outputs])[
        - CORAL logits: `logits["B N_c (K-1)"]` for $K$ ordinal classes.
        - Candidate score: $hat(s) in [0,1]$ from expected ordinal value.
        - Validity mask: `candidate_valid["B N_c"]` (inside EVL voxel bounds).
      ]
    ],
    [
      #color-block(title: [Decision rule])[
        Select the predicted best next view:
        $
          hat(bold(q)) = "argmax"_i hat(s)_i approx "argmax"_i "RRI"(bold(q)_i).
        $
      ]
      #quote-block(color: rgb("#285f82"))[
        EVL is frozen. Only pose-encoder + scorer head are trained.
      ]
    ],
  )
]

#slide(title: [EVL Feature Contract (frozen)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Why the 3D neck features?])[
        - Stable, information-rich voxel representation.
        - Avoids bottlenecking VIN through task-specific head predictions.
        - Efficient to query for many candidates (pool + sampling).
      ]
      #color-block(title: [Feature tensors we use])[
        - `neck/occ_feat["B Cocc D H W"]` (geometry/context)
        - `neck/obb_feat["B Cobb D H W"]` (optional semantics/context)
        - `voxel/T_world_voxel: PoseTW["B 12"]` (world$<-$voxel)
        - `voxel_extent["B 6"]` (voxel-frame AABB in metres)
      ]
    ],
    [
      #color-block(title: [Implementation location])[
        - Single adapter: `oracle_rri/oracle_rri/vin/backbone_evl.py`
        - Returns a minimal \"feature contract\" so VIN stays insulated from upstream #EVL input/output changes.
      ]
      #quote-block[Keep all EFM key expectations inside the EVL adapter; patch one place if schema changes.]
    ],
  )
]

#slide(title: [VIN: Descriptor + Encodings (shell_sh)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Descriptor (rig frame) + shapes])[
        From the training-aligned pose `T_camera_rig` (camera$<-$rig) we invert:
        $
          bold(T)_("rig"<-"cam") = (bold(T)_("cam"<-"rig"))^(-1).
        $

        Define:
        $
          r = ||bold(t)||, \
          bold(u) = bold(t)/(r + epsilon), \
          bold(f) = bold(R) bold(z)_("cam"), \
          s = <bold(f),-bold(u)>.
        $

        Shapes (symbols):
        - $bold(t)$: `(B, N_c, 3)`, $r$: `(B, N_c, 1)`
        - $bold(u)$: `(B, N_c, 3)`, $bold(f)$: `(B, N_c, 3)`, $s$: `(B, N_c, 1)`
      ]
      #color-block(title: [Which encoding for which feature?])[
        Encodings (default):
        - SH($L$): $bold(u)$ and $bold(f)$ (directions on $bb(S)^2$).
        - 1D Fourier: $r$ (scalar radius in metres).
        - Scalar MLP: $s$ (and optional extra scalars).

        Pose embedding:
        $
          bold(E)_("pose") = [bold(E)_u, bold(E)_f, bold(E)_r, bold(E)_s] in bb(R)^(B,N_c,d_"pose").
        $

        Note: We do not use $log("SO"(3))$ / `so3log` here.
      ]
    ],
    [
      #quote-block(color: rgb("#285f82"))[
        If candidates use `view_direction_mode=radial_away`, then $bold(f)$ often aligns with $bold(u)$ (so $s approx -1$).
      ]
    ],
  )
]

#slide(title: [VIN Head: Feature Aggregation + Scoring])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Scene features from EVL])[
        For each snippet:
        - Run frozen #EVL once #sym.arrow.r `occ_feat`, `obb_feat`, `T_world_voxel`, `voxel_extent`.
        - Global context: mean + max pool over $(D,H,W)$ (occ and optionally obb).
      ]
      #color-block(title: [Candidate-conditioned query (current baseline)])[
        - Local query: sample voxel features at the candidate camera center.
        - Define `candidate_valid` if the camera center is inside voxel extent.
        - Concatenate:
          + `E_pose` (SH + radius FF + scalars)
          + `E_global` (pooled voxel context)
          + `E_local` (sampled voxel features)
        - MLP #sym.arrow.r `CoralLayer` #sym.arrow.r logits.
      ]
    ],
    [
      #color-block(title: [Planned v0.2 upgrade: frustum query])[
        Center sampling is a weak view descriptor. Next step:
        - Sample $K$ points along a small ray set in front of the camera
        - Transform to voxel coordinates and sample features at those points
        - Pool (mean+max) with strict point-level validity masking
      ]
      #quote-block[
        This approximates \"what the candidate would see\" without rendering, and stays compatible with EVL's voxel
        feature interface.
      ]
    ],
  )
]

#slide(title: [VIN: Shell Descriptor (concept + statistics)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #figure(
        image(fig_path + "impl/vin/vin_shell_descriptor_concept.png", width: 98%),
        caption: [Conceptual descriptor for one candidate: $bold(t)$, $r=||bold(t)||$, $bold(u)$, $bold(f)$.],
      )
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_pose_descriptor.png", width: 98%),
        caption: [Empirical stats (one snippet): $r$, $s=<bold(f),-bold(u)>$, and az/el of $bold(u)$ vs $bold(f)$.],
      )
    ],
  )
]

#slide(title: [VIN: Encoding Visualizations (SH + radius Fourier)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #figure(
        image(fig_path + "impl/vin/vin_sh_components.png", width: 98%),
        caption: [Real spherical harmonics components over directions on $bb(S)^2$ ($L=1$ shown).],
      )
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_radius_fourier_features.png", width: 98%),
        caption: [1D Fourier features for radius (linear input): example $sin(2 pi omega r)$ curves.],
      )
    ],
  )
]

#slide(title: [VIN: EVL Neck Feature Diagnostics (frozen backbone)])[
  #figure(
    image(fig_path + "impl/vin/vin_evl_features.png", width: 94%),
    caption: [Mean absolute feature magnitude for `neck/occ_feat` and `neck/obb_feat` (mid-slice + histogram).],
  )
]

#slide(title: [VIN: Model Summary (rich)])[
  #figure(
    image(fig_path + "impl/vin/vin_rich_summary.png", width: 92%),
    caption: [Model overview (trainable modules + parameter counts) generated with `oracle_rri.utils.rich_summary`.],
  )
]

#slide(title: [Ordinal Binning: Thresholds + Example])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.9cm,
    [
      #color-block(title: [Fit once: map scalar RRI #sym.arrow.r ordinal label])[
        We fit a single global set of edges $bold(e) in bb(R)^{K-1}$ in clipped z-score space:
        $
          z = ("rri" - mu_s) / (sigma_s + epsilon), \
          z_"clip" = tanh(z / tau), \
          y = sum_(j=1)^(K-1) bb(1)_(z_"clip" > e_j).
        $

        - Choose $bold(e)$ by quantiles of $z_"clip"$ (bins ~balanced).
        - Thresholds are global (not per-candidate-set) so scores remain comparable across snippets.
      ]
      #color-block(title: [Where it lives])[
        - `oracle_rri/oracle_rri/vin/rri_binning.py`
        - saved once (JSON) and reused during training/inference
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_rri_binning.png", width: 98%),
        caption: [Example: raw RRI, clipped z distribution with quantile edges, and resulting label histogram.],
      )
    ],
  )
]

#slide(title: [CORAL Loss + Scoring])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [CORAL outputs])[
        For $K$ ordinal classes, predict $(K-1)$ threshold logits:
        - $l_k$ parameterizes $P(y > k)$ via $sigma(l_k)$.
        - Level targets:
        $
          t_k = bb(1)_(y > k),\ k = 0,...,K-2.
        $
      ]
      #color-block(title: [Loss + expected score])[
        CORAL loss is a sum of binary cross-entropies:
        $
          cal(L)_"CORAL" = sum_{k=0}^{K-2} "BCE"(sigma(l_k), t_k).
        $
        Score for selection uses the expected ordinal value:
        $
          hat(y) = sum_{k=0}^{K-2} sigma(l_k), \
          hat(s) = hat(y) / (K - 1).
        $
      ]
    ],
    [
      #color-block(title: [Reference implementation])[
        We use the MIT-licensed upstream implementation:
        #link("https://raschka-research-group.github.io/coral-pytorch/")[coral-pytorch].
      ]
      #color-block(title: [Masking (must-do)])[
        Compute loss only for valid candidates:
        - inside voxel grid: `candidate_valid`
        - finite labels: `isfinite(rri)`
        - optionally: require sufficient valid depth hits (future refinement)
      ]
    ],
  )
]

#slide(title: [Training Step: End-to-end labeler #sym.arrow.r VIN #sym.arrow.r loss])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Data path (one optimizer step)])[
        1. Sample `EfmSnippetView` from `AseEfmDataset`.
        2. Run `OracleRriLabeler.run(sample)`:
          + candidates #sym.arrow.r depth renders #sym.arrow.r backprojection #sym.arrow.r oracle #RRI
        3. Convert #RRI to ordinal labels via fitted binner.
        4. VIN forward (EVL frozen, head trainable) #sym.arrow.r CORAL logits.
        5. Backprop CORAL loss over valid candidates only.
      ]
      #quote-block(color: rgb("#fc5555"))[
        Online oracle labels are expensive. The minimal script is a correctness/smoke test; real training should cache
        labels (depths/PCs or RRIs).
      ]
    ],
    [
      #color-block(title: [Minimal script (smoke test)])[
        ```bash
        cd oracle_rri
        uv run python scripts/train_vin.py \
          --fit-snippets 2 --max-steps 10 --max-candidates 8 --device auto
        ```
      ]
      #color-block(title: [Alignment detail])[
        Train on the exact rendered subset:
        - world poses: `label_batch.depths.poses`
        - pose descriptor: `label_batch.depths.camera.T_camera_rig`
        - reference: `label_batch.depths.reference_pose`
      ]
    ],
  )
]

#slide(title: [Open Questions (VIN design choices)])[
  #color-block(title: [Backbone features])[
    - Use only `occ_feat` or also `obb_feat` (semantics) for #RRI prediction?
    - Compress voxel channels (1x1x1 conv) before pooling/sampling?
  ]
  #color-block(title: [Candidate-conditioned query])[
    - Center sampling vs. frustum point sampling (rays/depths) vs. tiny CA over points?
  ]
  #color-block(title: [Pose encoding])[
    - SH degree $L$ (2 vs 3) and normalization choice; include camera *up* direction?
    - Which scalar terms help most: $<bold(f),-bold(u)>$, height, gravity alignment, etc.?
  ]
  #color-block(title: [Ordinal setup + training])[
    - $K$ bins (15 default): does more resolution help or just add label noise?
    - Metrics to optimize/report: Spearman rank corr, top-$k$ recall, calibration of predicted scores.
  ]
]

// ---------------------------------------------------------------------------
// Engineering progress (from docs/contents/todos.qmd)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Engineering Progress],
  subtitle: [Resolved issues and shipped tooling],
)[
  #text(size: 16pt)[Condensed from `docs/contents/todos.qmd` (resolved items + \"Previously observed issues\").]
]

#slide(title: [Resolved Issues (High Impact)])[
  #color-block(title: [App + pipeline refactor])[
    - Separated UI (Streamlit panels) from compute (`OracleRriLabeler`) for reuse in training/CLI.
    - Replaced untyped Streamlit caches with a typed `AppState` + explicit stage caches.
  ]
  #color-block(title: [Correctness + reproducibility])[
    - Stabilized candidate indexing: rendered subset tracked via `CandidateDepths.candidate_indices`.
    - Candidate PCs align with the GT mesh (PoseTW #sym.arrow.l.r PyTorch3D extrinsics fixed).
    - Candidate PCs match the rendered frusta (pixel-center #sym.arrow.r NDC backprojection fixed).
    - Depth histograms no longer mislead (miss pixels masked via `depths_valid_mask`).
  ]
]

#slide(title: [Rendering & Backprojection: Issues and Fixes])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Observed Issues])[
        - Backprojected points \"all lie in front\" of the reference pose.
        - Depth maps invalid for candidates that look away from the reference.
        - Frusta visualization didn't match the backprojected points.
        - \"Look-through-walls\" artefacts (points behind walls).
      ]
    ],
    [
      #color-block(title: [Root causes + fixes])[
        - *PoseTW #sym.arrow.l.r PyTorch3D extrinsics mismatch*:
          + Fix camera extrinsics so rendering/unprojection uses the same world #sym.arrow.r camera mapping as `PoseTW`.
        - *Wrong unprojection space*:
          + Convert pixel centers `(x+0.5, y+0.5)` to NDC (min-side scaling) and unproject with `from_ndc=true`.
        - *Histogram semantics*:
          + Mask miss pixels (`depths_valid_mask`) when computing hit ratios / histograms / backprojection.
        - *Rig-basis vs display twist*:
          + `rotate_yaw_cw90` is a fixed 90#sym.degree twist about the pose-local `+Z` (forward) axis.
          + Apply it once as a rig-basis correction for candidate generation (otherwise azimuth/elevation look swapped).
          + Do not apply it again in candidate plotting: double-rotation becomes obvious once roll jitter is enabled.
      ]
    ],
  )
]

// Bibliography
#slide(title: [Bibliography])[
  #bibliography("/references.bib", style: "/ieee.csl")
]

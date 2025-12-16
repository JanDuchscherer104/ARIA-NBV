// Oracle RRI pipeline slides (ASE → candidates → rendering → backprojection → scoring)
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
    subtitle: [Data → Candidates → Rendering → Backprojection → Metrics],
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
  title: [oracle_rri: Oracle RRI Labeling Pipeline],
  subtitle: [Reusable compute path for dashboard, preprocessing, and training],
)[
  #text(size: 16pt)[Stage-by-stage conceptual overview of `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.]
]

#slide(title: [Pipeline at a Glance])[
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
          + `OracleRRIConfig` (point↔mesh distances → #RRI)
        - Pipeline knob: `backprojection_stride` (pixel subsampling)
      ]
    ],
    [
      #color-block(title: [Stages])[
        1. *Candidate Generation*: sample + orient + prune `PoseTW["C 12"]`
        2. *Depth Rendering*: render `depths["C H W"]` (+ valid mask)
        3. *Backprojection*: depth hits → `CandidatePointClouds`
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

  #quote-block(color: rgb("#285f82"))[
    “Blank wall has high RRI” is not necessarily a rendering bug: with GT depth, newly observed wall surfaces can
    legitimately reduce point↔mesh distances even if real semi-dense SLAM would add few points on low-texture walls.
  ]
]

// ---------------------------------------------------------------------------
// Data handling (download → dataset → typed views → mesh processing)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Data Handling],
  subtitle: [Downloading ASE + streaming ATEK → typed `EfmSnippetView`],
)[
  #figure(image(fig_path + "scene-script/ase_modalities.jpg", width: 80%), caption: [@SceneScript-avetisyan2024])
]

#slide(title: [ASE Modalities Used by oracle_rri])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2cm,
    [
      #color-block(title: [Core modalities])[
        - *Trajectory*: `PoseTW` world←rig for each timestamp
        - *Semi-dense SLAM points*: sparse but informative geometry cues
        - *GT mesh*: watertight-ish surface geometry (oracle target)
        - *Camera calibration*: `CameraTW` intrinsics/extrinsics per stream
      ]
      #color-block(title: [Why we need all of them])[
        - Candidate poses must be sampled in a *physically consistent* world frame.
        - Rendering needs a mesh + camera model.
        - Backprojection needs rendered depth + the exact camera used to render.
        - RRI uses point↔mesh distances, so both point sets and mesh must share coordinates.
      ]
    ],
    [
      #color-block(title: [Typed views])[
        - `EfmCameraView`: images + `CameraTW` + timestamps
        - `EfmTrajectoryView`: `t_world_rig` + helpers (`final_pose()`)
        - `EfmPointsView`: `points_world`, `dist_std`, bounds, `lengths`
        - `EfmGTView` / `EfmObbView`: GT timestamps and optional entity boxes
      ]
      #quote-block[
        The dataset yields *views* over an EFM dictionary (zero-copy): move tensors with `.to(device)` without cloning.
      ]
    ],
  )
]

#slide(title: [Unified ASE Downloader CLI])[
  #color-block(title: [What it does])[
    - One entrypoint downloads GT meshes + ATEK shards with consistent filtering.
    - Mesh download: SHA1 validation + ZIP extraction to `.ply`.
    - ATEK download: write filtered JSON for `download_atek_wds_sequences`.
  ]
  #color-block(title: [Modes])[
    - `download`: pull meshes and/or ATEK for top-N scenes (cap snippets, prefer richest scenes first).
    - `list`: show scenes with GT meshes and snippet counts (sorted by snippet density).
  ]
  #color-block(title: [Usage])[
    #text(size: 13pt)[
      - `python -m oracle_rri.data.downloader list --n=8`
      - `python -m oracle_rri.data.downloader download --n_scenes=5 --max_snippets=2 --c=efm_eval`
      - Flags: `--skip_meshes`, `--skip_atek`, `--prefer_scenes_with_max_snippets`, `--output-dir`.
    ]
  ]
]

#slide(title: [AseEfmDataset + Mesh Handling])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Dataset wrapper])[
        - `AseEfmDataset` wraps `load_atek_wds_dataset_as_efm` → yields `EfmSnippetView`.
        - Resolves tar shards from `PathConfig`, infers scene/snippet ids.
        - Optional mesh loading + simplification + caching (`mesh_cache.py`).
      ]
      #color-block(title: [Mesh processing])[
        - Load `.ply` via `trimesh` (vertices + faces).
        - Optional quadric simplification for speed.
        - Optional cropping to occupancy bounds for stability/efficiency.
      ]
    ],
    [
      #color-block(title: [Typed access in practice])[
        ```python
        from oracle_rri.data import AseEfmDatasetConfig

        cfg = AseEfmDatasetConfig(
            scene_ids=["81048"],
            atek_variant="efm_eval",
            load_meshes=True,
        )
        ds = cfg.setup_target()
        sample = next(iter(ds))

        rgb = sample.camera_rgb
        traj = sample.trajectory
        sem = sample.semidense
        print(sample.has_mesh, sample.mesh is not None)
        ```
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Candidate generation (SE(3) poses around a reference pose)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Candidate Generation],
  subtitle: [Generate and prune candidate `PoseTW` around a reference pose],
)[
  #text(size: 16pt)[Implemented in `oracle_rri/oracle_rri/pose_generation/`.]
]

#slide(title: [CandidateViewGenerator: Overview])[
  #color-block(title: [Inputs])[
    - Reference pose: last rig/camera pose (optionally gravity-aligned).
    - Occupancy extent (AABB) from semidense bounds or mesh bounds.
    - GT mesh (for clearance + collision pruning).
  ]
  #color-block(title: [Outputs])[
    - Candidate poses: `PoseTW["C 12"]` (world ← cam)
    - Diagnostics: per-rule masks + optional debug stats
    - Stable indexing: map back to global candidate indices for UI alignment
  ]
  #color-block(title: [Design goals])[
    - Deterministic sampling via `seed`
    - Modular pruning rules (`Rule` protocol)
    - Clear separation: sampling (positions), orientation, pruning
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
          - *Forward-biased*: PowerSpherical / von-Mises–Fisher concentration (`kappa`) around forward axis.
          - *Gravity alignment*: remove pitch/roll in the sampling frame (`align_to_gravity=True`).

          #v(0.4em)
          *Intuition*
          - Explore around the last pose while respecting the rig’s reachable neighborhood.
          - Limit elevation to remain “human-like” (no ceiling cams).
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
          For camera center `p` and target `t`:
          $
            bold(z) & = (bold(t) - bold(p)) / (||bold(t) - bold(p)||) \
            bold(y) & = bold(u)_"world" \
            bold(x) & = (bold(y) times bold(z)) / (||bold(y) times bold(z)||).
          $
          Then re-orthogonalize: $bold(y) = bold(z) times bold(x)$.
        ]
      ],
      [
        #color-block(title: [Implementation notes])[
          - Implemented via `OrientationBuilder` in `orientations.py`.
          - View jitter is applied as right-multiplicative deltas (keeps bases orthonormal).
          - `align_to_gravity=True` uses a gravity-aligned reference pose before sampling.
        ]
        #quote-block(color: rgb("#285f82"))[
          *Why roll-free?* Roll adds little NBV value but makes collision-free shells and visual debugging harder.
        ]
      ],
    )
  ]
]

#slide(title: [Rule: Min Distance to Mesh])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Criterion])[
        For candidate center $bold(c)_i$ and mesh $bold(cal(M))$:
        $
          d_i = min_{bold(x) in bold(cal(M))} ||bold(c)_i - bold(x)||_2.
        $
        Keep iff $d_i > d_"min"$.
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Reject viewpoints that start inside/too close to GT geometry.
        - Prevents unstable renders (near-plane clipping explosions).
        - Implemented via PyTorch3D point↔mesh distance or trimesh proximity queries.
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
        Straight-line segment from origin $bold(o)$ to candidate center $bold(p)_i$:
        $
          bold(r)_i(t) = bold(o) + t hat(bold(d)_i),
          t in [0, ||bold(p)_i - bold(o)||].
        $
        Reject if the segment intersects the mesh (optionally with clearance).
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Enforce collision-free translation in a simple, explainable way.
        - Backends:
          + PyTorch3D distance sampling along the segment
          + Trimesh / PyEmbree ray intersector (faster when available)
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
          x_"min" <= x_i <= x_"max",
          y_"min" <= y_i <= y_"max",
          z_"min" <= z_i <= z_"max".
        $
      ]
    ],
    [
      #color-block(title: [Intuition])[
        - Prevent candidates from leaking outside the local room/scene bounds.
        - Bounds come from semidense metadata (`volume_min/max`) or mesh crop bounds.
      ]
    ],
  )
]

#slide(title: [Pose Generation Diagram])[
  #figure(image("pose_generation_diagram.png", width: 90%), caption: [oracle_rri.pose_generation subpackage])
]

// ---------------------------------------------------------------------------
// Depth rendering (simulate candidate observation)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Depth Rendering],
  subtitle: [Render candidate depth maps from the GT mesh],
)[
  #text(size: 16pt)[Implemented in `oracle_rri/oracle_rri/rendering/`.]
]

#slide(title: [CandidateDepthRenderer: Responsibilities])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Candidate selection])[
        - Render only a subset for performance:
          + oversample `oversample_factor`
          + cap to `max_candidates_final`
        - Keep stable indices via `candidate_indices` (maps into the full candidate list).
      ]
      #color-block(title: [Outputs])[
        - `depths["C H W"]` in metres (metric z-depth)
        - `depths_valid_mask["C H W"]` (hit + near/far clipping)
        - `poses` (world ← cam) and `p3d_cameras` for later backprojection
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
      #quote-block[
        *Practical tip*: Always plot depth *with* the valid mask applied; otherwise invalid pixels dominate histograms.
      ]
    ],
  )
]

#slide(title: [PyTorch3D Rasterization (Conceptual)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Camera model])[
        We use PyTorch3D `PerspectiveCameras` in *screen space* (`in_ndc=false`):
        $
          u = f_x X/Z + c_x,
          v = f_y Y/Z + c_y.
        $
        The renderer internally converts screen coordinates to NDC using
        $s = min(H, W)$ (important for non-square images).
      ]
      #color-block(title: [Rasterization output])[
        `MeshRasterizer` returns fragments:
        - `pix_to_face`: face index per pixel (hit/miss)
        - `zbuf`: per-pixel z-depth (metres when configured consistently)
        - optional barycentric coordinates (not used for oracle depth)
      ]
    ],
    [
      #color-block(title: [Coordinate conventions])[
        - PyTorch3D NDC: +X left, +Y up, right-handed.
        - We keep all core geometry in the physical Aria LUF world frame.
        - Extrinsics must be consistent with `PoseTW` conventions when building `PerspectiveCameras(R,T)`.
      ]
      #color-block(title: [Why rasterization?])[
        - GPU batch rendering for many candidates
        - Deterministic depth (z-buffer)
        - Works directly with mesh tensors (`verts`, `faces`)
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Backprojection (depth hits → world-frame point clouds)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Backprojection],
  subtitle: [Depth hits → candidate point clouds (world frame)],
)[
  #text(size: 16pt)[Implemented in `unproject.py` and `candidate_pointclouds.py`.]
]

#slide(title: [Depth → 3D Points (Math)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Pixel → NDC mapping (PyTorch3D)])[
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
      #color-block(title: [Why NDC here?])[
        Using NDC consistently matches the internal camera transform used by the rasterizer.
        This avoids subtle “frustum ≠ points” bugs for non-square images.
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
        Stride is a key compute knob: smaller stride = denser candidate PCs, but more GPU work in both unprojection and point↔mesh distances.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Oracle RRI (point↔mesh distances + normalization)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Oracle RRI Metric],
  subtitle: [Point↔mesh distances (accuracy + completeness) → normalized improvement],
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
    - $d(.,.)$: *bidirectional* point↔mesh distance
  ]

  #quote-block(color: rgb("#285f82"))[
    The ratio cancels units and makes scores comparable across scenes and scales.
  ]
]

#slide(title: [Point↔Mesh Distance: Accuracy + Completeness])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Accuracy (P → M)])[
        Average distance from reconstruction points to the GT surface:
        $
          d_{bold(cal(P)) -> bold(cal(M))} =
          1/(|bold(cal(P))|) sum_{bold(p) in bold(cal(P))} min_{bold(x) in bold(cal(M))} ||bold(p) - bold(x)||_2^2.
        $
        - Detects *over-reconstruction* (spurious geometry).
        - Implemented via PyTorch3D `point_face_distance` (point→triangle).
      ]
    ],
    [
      #color-block(title: [Completeness (M → P)])[
        Average distance from GT surface elements to the reconstruction:
        $
          d_{bold(cal(M)) -> bold(cal(P))} approx 1/(|bold(cal(F))|) sum_{bold(f) in bold(cal(F))} min_{bold(p) in bold(cal(P))} d(bold(f), bold(p))^2.
        $
        - Detects *under-reconstruction* (missing geometry).
        - Implemented via PyTorch3D `face_point_distance` (triangle→point).
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
        - point→face: `(P_tot,)` (each point to nearest triangle)
        - face→point: `(F_tot,)` (each face to nearest point)
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

#slide(title: [VIN: I/O Formulation + Goal])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Inputs])[
        - *State*: raw EFM snippet dict `efm: dict[str, Any]` (fed to #EVL).
        - *Candidates*: `PoseTW["N 12"]` world←camera.
        - *Reference*: last rig pose in snippet (or explicit override).
        - Training alignment: use `OracleRriLabelBatch.depths.camera.T_camera_rig` (camera←reference).
      ]
      #color-block(title: [Outputs])[
        - Ordinal logits: `logits["N (K-1)"]` (CORAL thresholds).
        - Score for selection: expected class $\hat(y)$ (normalized to $[0,1]$).
        - Validity mask: `candidate_valid["N"]` (inside #EVL voxel bounds).
      ]
    ],
    [
      #color-block(title: [Decision rule])[
        Select the predicted best next view:
        $
          hat(bold(q)) = "argmax"_i hat(s)_i approx "argmax"_i "RRI"(bold(q)_i).
        $
        where $hat(s)_i$ is the normalized expected ordinal value derived from CORAL logits.
      ]
      #quote-block(color: rgb("#285f82"))[
        Key idea: amortize “candidate → render → distance → #RRI” with a learned head that can score many candidates
        cheaply, while keeping #EVL frozen as a strong scene prior (@EFM3D-straub2024).
      ]
    ],
  )
]

#slide(title: [Frozen EVL Backbone: Feature Contract])[
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
        - `neck/occ_feat["B Cocc D H W"]` (occupancy neck volume)
        - `neck/obb_feat["B Cobb D H W"]` (optional semantics / detection neck)
        - `voxel/T_world_voxel: PoseTW["B 12"]`
        - `voxel_extent["B 6"]` = $(x_min,x_max,y_min,y_max,z_min,z_max)$ in voxel frame (metres)
      ]
    ],
    [
      #color-block(title: [Implementation location])[
        - Single adapter: `oracle_rri/oracle_rri/vin/backbone_evl.py`
        - Returns a minimal “feature contract” so VIN stays insulated from upstream #EVL input/output changes.
      ]
      #quote-block[
        Practical stability rule: keep all EFM key expectations inside the EVL adapter; patch one place if schema
        changes.
      ]
    ],
  )
]

#slide(title: [Candidate Pose Descriptor (Shell-aware, in reference frame)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Where it comes from (training)])[
        Oracle labels are computed for a rendered subset, so VIN must be trained on the *same* candidates:
        - `label_batch.depths.poses` (world←camera, rendered subset)
        - `label_batch.depths.camera.T_camera_rig` (camera←reference rig, same subset)

        We derive the descriptor from:
        $
          bold(T)_("rig"<-"cam") = (bold(T)_("cam"<-"rig"))^-1.
        $
      ]
      #color-block(title: [Shell descriptor])[
        Let $bold(T)_("rig"<-"cam") = (bold(R), bold(t))$, with $bold(z)_("cam") = (0,0,1)$ (LUF forward). Compute:
        $
          r = ||bold(t)||, \
          bold(u) = bold(t)/(r + epsilon) in bb(S)^2, \
          bold(f) = bold(R) bold(z)_("cam") in bb(S)^2.
        $
        Cheap scalar inductive bias (example):
        $
          s = <bold(f), -bold(u)>.
        $
      ]
    ],
    [
      #quote-block(color: rgb("#285f82"))[
        Why shell-aware? Candidate generation is a spherical (or biased-spherical) sampling process. Encoding directions
        on the sphere explicitly matches that prior.
      ]
      #color-block(title: [Interpretation (intuition)])[
        - $bold(t)$: candidate camera center in reference rig frame.
        - $r$: how far we move (translation budget).
        - $bold(u)$: *where* we move on the shell (direction).
        - $bold(f)$: *where we look* (candidate optical axis).
        - $s=<bold(f),-bold(u)>$: “look back at reference” score ($s≈1$ means looking toward the reference).
      ]
      #color-block(title: [No so3log])[
        We avoid $log("SO"(3))$ / `so3log` mappings for the default VIN path. The descriptor uses *directions* and a
        small set of scalars instead.
      ]
    ],
  )
]

#slide(title: [Pose Encodings: SH (directions) + 1D Fourier (radius) + LFF baseline])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Spherical harmonics for $u,f in bb(S)^2$])[
        - Encode $bold(u)$ and $bold(f)$ with *real* spherical harmonics up to degree $L$.
        - SH feature dim: $(L+1)^2$ (per direction).
        - Project each SH vector to a learnable embedding (small MLP).

        Implementation: `oracle_rri/oracle_rri/vin/spherical_encoding.py` using
        #link("https://github.com/e3nn/e3nn")[e3nn].
      ]
      #color-block(title: [1D Fourier features for radius])[
        SH is defined on directions, not on the scalar radius. We encode
        $r$ via 1D Fourier features (default on $log(r+epsilon)$):
        $
          phi(s) = [sin(2 pi bold(B) s), cos(2 pi bold(B) s)],
          s = log(r + epsilon).
        $
        then project to a learnable radius embedding and concatenate.
      ]
    ],
    [
      #color-block(title: [Baseline: Learnable Fourier Features (LFF)])[
        Optional baseline encoding (no SH):
        - build a 6D pose vector $bold(x) = [bold(t), bold(f)] in bb(R)^6$
        - learn a projection + sinusoidal features + MLP (LFF)

        Code: `oracle_rri/oracle_rri/vin/pose_encoding.py` (based on
        #link("https://github.com/JHLew/Learnable-Fourier-Features")[Learnable-Fourier-Features]).
      ]
      #color-block(title: [Which features get which encoding?])[
        - SH: $bold(u)$ (position direction), $bold(f)$ (forward direction)
        - 1D Fourier: $log(r+epsilon)$
        - Scalar MLP: $s = <bold(f),-bold(u)>$ (and optional extras later)
        - LFF baseline: $[bold(t),bold(f)]$ (6D), *not* `so3log(R)`
      ]
    ],
  )
]

#slide(title: [Radius Encoding: $r$ vs $log(r+epsilon)$])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.9cm,
    [
      #color-block(title: [Does $log(r)$ help for $r in [0.6, 1.8]$ m?])[
        For our current shell radii $r in [0.6, 1.8]$ m (a narrow range), $log(r)$ mainly:
        - compresses the upper tail slightly,
        - changes the effective frequency content of 1D Fourier features.

        It is *not* essential here; using $r$ directly is a valid alternative.
      ]
      #color-block(title: [Recommendation])[
        - Keep a config switch: encode either $r$ or $log(r+epsilon)$.
        - Prefer $r$ when the radius range is narrow and fixed.
        - Prefer $log(r+epsilon)$ when candidate radii vary strongly across scenes / policies.
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_radius_fourier_features.png", width: 98%),
        caption: [Fourier feature curves differ significantly between linear and log input.],
      )
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
        - Run frozen #EVL once → `occ_feat`, `obb_feat`, `T_world_voxel`, `voxel_extent`.
        - Global context: mean + max pool over $(D,H,W)$ (occ and optionally obb).
      ]
      #color-block(title: [Candidate-conditioned query (current baseline)])[
        - Local query: sample voxel features at the candidate camera center.
        - Define `candidate_valid` if the camera center is inside voxel extent.
        - Concatenate:
          + `E_pose` (SH + radius FF + scalars)
          + `E_global` (pooled voxel context)
          + `E_local` (sampled voxel features)
        - MLP → `CoralLayer` → logits.
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
        This approximates “what the candidate would see” without rendering, and stays compatible with EVL’s voxel
        feature interface.
      ]
    ],
  )
]

#slide(title: [VIN: Input Features (real snippet, shapes)])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [EFM snippet inputs (key tensors)])[
        Generated via `oracle_rri/scripts/summarize_vin.py` on scene `81283` (device: `cuda`, $N=4$ candidates).
        - `rgb/img`: `Tensor(20, 3, 240, 240)`
        - `slaml/img`: `Tensor(20, 1, 240, 320)`
        - `slamr/img`: `Tensor(20, 1, 240, 320)`
        - `pose/t_world_rig`: `PoseTW(20, 12)`
        - `points/p3s_world`: `Tensor(20, 50000, 3)`
        - `points/dist_std`: `Tensor(20, 50000)`
        - `pose/gravity_in_world`: `Tensor(3,)`
      ]
      #color-block(title: [Pose descriptor + VIN concat])[
        Derived descriptor in reference frame:
        - $bold(t)$: `(1, 4, 3)`, $r=||bold(t)||$: `(1, 4, 1)`
        - $bold(u)$: `(1, 4, 3)`, $bold(f)$: `(1, 4, 3)`
        - $s = <bold(f),-bold(u)>$: `(1, 4, 1)`

        VIN head input features:
        - `E_pose`: `(1, 4, 128)`
        - `E_global`: `(1, 4, 256)`
        - `E_local`: `(1, 4, 128)`
        - concat: `(1, 4, 512)`
      ]
    ],
    [
      #color-block(title: [EVL feature contract])[
        - `occ_feat`: `(1, 64, 48, 48, 48)`
        - `obb_feat`: `(1, 64, 48, 48, 48)`
        - `T_world_voxel`: `PoseTW(1, 12)`
        - `voxel_extent`: `(6,)`
      ]
      #quote-block[
        EVL is frozen; only the VIN pose encoder + scorer head are trained.
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
        caption: [1D Fourier features for radius: compare $r$ vs $log(r+epsilon)$ as input.],
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

#slide(title: [VIN: torchsummary (ShellShPoseEncoder)])[
  #color-block(title: [ShellShPoseEncoder (SH + 1D Fourier radius)])[
    #text(size: 12pt)[
      ```text
      Layer (type:depth-idx)                   Output Shape              Param #
      ├─Sequential: 1-1                        [-1, 4, 32]               --
      │    ├─Linear: 2-1                       [-1, 4, 32]               544
      │    ├─GELU: 2-2                         [-1, 4, 32]               --
      │    └─Linear: 2-3                       [-1, 4, 32]               1,056
      ├─Sequential: 1-2                        [-1, 4, 32]               --
      │    ├─Linear: 2-4                       [-1, 4, 32]               544
      │    ├─GELU: 2-5                         [-1, 4, 32]               --
      │    └─Linear: 2-6                       [-1, 4, 32]               1,056
      ├─FourierFeatures: 1-3                   [-1, 4, 17]               8
      ├─Sequential: 1-4                        [-1, 4, 32]               --
      │    ├─Linear: 2-7                       [-1, 4, 32]               576
      │    ├─GELU: 2-8                         [-1, 4, 32]               --
      │    └─Linear: 2-9                       [-1, 4, 32]               1,056
      ├─Sequential: 1-5                        [-1, 4, 32]               --
      │    ├─Linear: 2-10                      [-1, 4, 64]               128
      │    ├─GELU: 2-11                        [-1, 4, 64]               --
      │    └─Linear: 2-12                      [-1, 4, 32]               2,080

      Total params: 7,048
      Trainable params: 7,048
      ```
    ]
  ]
]

#slide(title: [VIN: torchsummary (VinScorerHead)])[
  #color-block(title: [VinScorerHead (MLP + CORAL)])[
    #text(size: 12pt)[
      ```text
      Layer (type:depth-idx)                   Output Shape              Param #
      ├─Sequential: 1-1                        [-1, 256]                 --
      │    ├─Linear: 2-1                       [-1, 256]                 131,328
      │    ├─GELU: 2-2                         [-1, 256]                 --
      │    ├─Linear: 2-3                       [-1, 256]                 65,792
      │    └─GELU: 2-4                         [-1, 256]                 --
      ├─CoralLayer: 1-2                        [-1, 14]                  --
      │    └─CoralLayer: 2-5                   [-1, 14]                  --
      │         └─Linear: 3-1                  [-1, 1]                   256

      Total params: 197,376
      Trainable params: 197,376
      ```
    ]
  ]
]

#slide(title: [Ordinal Binning: Thresholds for K classes])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Why bin RRI?])[
        Raw #RRI is noisy and can be hard to regress directly.
        VIN-NBV discretizes #RRI into ordinal bins and learns with a ranking-aware loss (CORAL).
      ]
      #color-block(title: [Binning procedure (RriOrdinalBinner)])[
        Fit once on a bootstrap set of oracle labels:
        1. Choose stage ids (baseline: all stage=0).
        2. Compute per-stage mean/std and z-normalize:
        $
          z = ("rri" - mu_s) / (sigma_s + epsilon).
        $
        3. Soft-clip: $z_"clip" = tanh(z / tau)$.
        4. Choose bin edges by quantiles so bins have ~equal mass.
        5. Save binner (edges + stats) to JSON and reuse during training.
      ]
    ],
    [
      #color-block(title: [Practical notes])[
        - Quantile edges reduce class imbalance by construction.
        - Thresholds are global (not per-candidate-set), so scores stay comparable across snippets.
        - In the minimal script: binner is fit online and saved as `rri_binner.json`.
      ]
      #color-block(title: [Where it lives])[
        - `oracle_rri/oracle_rri/vin/rri_binning.py`
        - used in `oracle_rri/scripts/train_vin.py`
      ]
    ],
  )
]

#slide(title: [Ordinal Binning: Example fit on oracle labels])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.9cm,
    [
      #color-block(title: [What the thresholds represent])[
        We fit a single global set of edges $bold(e) in bb(R)^{K-1}$ in clipped z-score space:
        $
          z = ("rri" - mu_s) / (sigma_s + epsilon), \
          z_"clip" = tanh(z / tau), \
          y = sum_(j=1)^(K-1) bb(1)_(z_"clip" > e_j).
        $

        Quantile fitting makes bins approximately balanced (good for CORAL training stability).
      ]
      #color-block(title: [Why not “rank within a candidate set”?])[
        We need *absolute* labels so scores are comparable across snippets:
        - per-set ranks destroy cross-scene calibration,
        - selection uses $hat(s)$ across candidates, not across batches.
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

#slide(title: [Training Step: End-to-end labeler → VIN → loss])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Data path (one optimizer step)])[
        1. Sample `EfmSnippetView` from `AseEfmDataset`.
        2. Run `OracleRriLabeler.run(sample)`:
          + candidates → depth renders → backprojection → oracle #RRI
        3. Convert #RRI to ordinal labels via fitted binner.
        4. VIN forward (EVL frozen, head trainable) → CORAL logits.
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
    - Should we compress voxel channels (1×1×1 conv) before pooling/sampling?
  ]
  #color-block(title: [Candidate-conditioned query])[
    - Center sampling vs. frustum point sampling (rays/depths) vs. tiny attention over points?
    - How strict should validity masking be (depth-hit count, voxel bounds, free-space)?
  ]
  #color-block(title: [Pose encoding])[
    - SH degree $L$ (2 vs 3) and normalization choice; include camera *up* direction?
    - Radius encoding: $r$ vs $log(r+epsilon)$; number of Fourier frequencies; learnable vs fixed $bold(B)$.
    - Which scalar terms help most: $<bold(f),-bold(u)>$, height, gravity alignment, etc.?
  ]
  #color-block(title: [Ordinal setup + training])[
    - $K$ bins (15 default): does more resolution help or just add label noise?
    - Stage definition for stage-aware normalization: how to define “stage” in ASE snippets?
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
  #text(size: 16pt)[Condensed from `docs/contents/todos.qmd` (resolved items + “Previously observed issues”).]
]

#slide(title: [Resolved Issues (High Impact)])[
  #color-block(title: [App + pipeline refactor])[
    - Separated UI (Streamlit panels) from compute (`OracleRriLabeler`) for reuse in training/CLI.
    - Replaced untyped Streamlit caches with a typed `AppState` + explicit stage caches.
  ]
  #color-block(title: [Correctness + reproducibility])[
    - Stabilized candidate indexing: rendered subset tracked via `CandidateDepths.candidate_indices`.
    - Candidate PCs align with the GT mesh (PoseTW↔PyTorch3D extrinsics fixed).
    - Candidate PCs match the rendered frusta (pixel-center→NDC backprojection fixed).
    - Depth histograms no longer mislead (miss pixels masked via `depths_valid_mask`).
  ]
]

#slide(title: [Rendering & Backprojection: Issues and Fixes])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Symptoms we observed])[
        - Backprojected points “all lie in front” of the reference pose.
        - Depth maps invalid for candidates that look away from the reference.
        - Frusta visualization didn’t match the backprojected points.
        - “Look-through-walls” artefacts (points behind walls).
      ]
    ],
    [
      #color-block(title: [Root causes + fixes])[
        - *PoseTW ↔ PyTorch3D extrinsics mismatch*:
          + Fix camera extrinsics so rendering/unprojection uses the same world→camera mapping as `PoseTW`.
        - *Wrong unprojection space*:
          + Convert pixel centers `(x+0.5, y+0.5)` to NDC (min-side scaling) and unproject with `from_ndc=true`.
        - *Histogram semantics*:
          + Mask miss pixels (`depths_valid_mask`) when computing hit ratios / histograms / backprojection.
        - *Aria UI rotation*:
          + `rotate_yaw_cw90` is visual-only; do not apply it to physical geometry.
      ]
    ],
  )
]

// Bibliography
#slide(title: [Bibliography])[
  #bibliography("/references.bib", style: "/ieee.csl")
]

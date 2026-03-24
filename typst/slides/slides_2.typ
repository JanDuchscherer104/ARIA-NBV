// This file defines slides for the ASE, EFM3D & EVL talk.
// It is based on the definitely-not-isec-theme provided in the skeleton.

#import "@preview/definitely-not-isec-slides:1.0.1": *
#import "@preview/muchpdf:0.1.1": muchpdf

// Import shared macros and symbols
#import "../shared/macros.typ": *

#let fig_path = "../../figures/"

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: false, // Disable to avoid conflicts with page numbers
  institute: [HM],
  logo: [#image(fig_path + "hm-logo.svg", width: 2cm)],
  config-info(
    title: [ASE, EFM3D & EVL: Datasets, Models & Tools for NBV],
    subtitle: [Towards Relative Reconstruction Metrics for Next-Best-View],
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
  config-common(
    handout: false,
  ),
  config-colors(
    primary: rgb("fc5555"),
  ),
)

// Set global text size
#set text(size: 17pt)

// Style links to be blue and underlined
#show link: set text(fill: blue)
#show link: it => underline(it)

// The title slide summarises the talk.
#title-slide()

// ASE Section
#section-slide(title: [Aria Synthetic Environments], subtitle: [Dataset for Egocentric 3D Scene Understanding])[
  #figure(image(fig_path + "scene-script/ase_modalities.jpg", width: 80%), caption: [@SceneScript-avetisyan2024])
]

// ASE overview slide
#slide(title: [ASE Dataset Overview])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5cm,
    [#text(size: 16pt)[
      #color-block(title: [Dataset Content])[
        - 100,000 unique multi-room interior scenes
        - \~2-min egocentric trajectories per scene
        - Populated with ~8,000 3D objects
        - Aria camera & lens characteristics
      ]

      #color-block(title: [Ground Truth Annotations])[
        - #emph-color[#SixDoF] trajectories
        - RGB-D frames
        - 2D panoptic segmentation
        - Semi-dense #SLAM #PC w/ visibility info
        - 3D floor plan (#SSL format)
        - #emph-it[GT meshes] as #code-inline[.ply] files
      ]

      #v(0.3em)
      *Key Resources*
      - Project Aria Tools for data access
      - #link(
          "https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset",
        )[ASE documentation] @ProjectAria-ASE-2025 @SceneScript-avetisyan2024
    ]],
    [
      #v(1em)
      #figure(
        [#image(fig_path + "scene-script/ase_primitives.jpg", width: 90%)
          #v(0.5em)
          #image(fig_path + "efm3d/gt_mesh.jpg", width: 90%)],
        caption: [@SceneScript-avetisyan2024],
      )
    ],
  )
]

// ASE Dataset Structure
#slide(title: [ASE Dataset Structure])[
  #text(size: 10pt)[
    ```
    scene_id/
    |-- ase_scene_language.txt          # Ground truth scene layout in SSL format
    |-- object_instances_to_classes.json # Mapping from instance IDs to semantic classes
    |-- trajectory.csv                   # 6DoF camera poses along the egocentric path
    |-- semidense_points.csv.gz          # Semi-dense 3D point cloud from MPS SLAM
    |-- semidense_observations.csv.gz    # Point observations (which images see which points)
    |-- rgb/                             # RGB image frames
    |   |-- 000000.png
    |   |-- ...
    |-- depth/                           # Ground truth depth maps
    |   |-- 000000.png
    |   |-- ...
    |-- instances/                       # Instance segmentation masks
        |-- 000000.png
        |-- ...
    ```
  ]
]

// EFM3D Section
#section-slide(title: [EFM3D Benchmark], subtitle: [3D Egocentric Foundation Model: \ Egocentric Voxel Lifting (EVL)
  #figure(
    muchpdf(read(fig_path + "efm3d/EFM3D_teaser_v1.pdf", encoding: none), width: 27cm),
    caption: [@EFM3D-straub2024],
  )
])

#slide(title: [EFM3D & EVL])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [EFM3D Tasks])[
        - 3D object detection
        - 3D surface regression (occupancy volumes)
          - on #ASE, #ADT#footnote[#text(size: 12pt)[Aria Digital Twin]], #AEO#footnote[#text(size: 12pt)[Aria Everyday Objects: small-scale, real-world w/ 3D OBBs]] datasets
      ]

      #color-block(title: [EVL Architecture])[
        - Utilizes #emph-bold[all] available egocentric modalities:
          + multiple (rectified) RGB, grayscale, and semi-dense points inputs
          + camera intrinsics and extrinsics
        - #emph-color[16.7M trainable] + 86.6M frozen params
        - Inherits foundational capabilities from frozen 2D model (DinoV2.5) by lifting 2D features to 3D @EFM3D-straub2024
      ]
    ],
    [
      #v(0.5em)
      #figure(muchpdf(read(fig_path + "efm3d/efm3d_arch_v1.pdf", encoding: none)), caption: [@EFM3D-straub2024])
    ],
  )
]

// EVL Architecture Details
#slide(title: [EVL: Egocentric Voxel Lifting Architecture])[
  #color-block(title: [Model Overview])[
    *Egocentric Voxel Lifting (EVL)*: Multi-task 3D perception from egocentric video

    #emph-bold[Key Principle]: Lift 2D image features to 3D voxel space using camera geometry
  ]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Input Formulation])[
        $ bold(X)_"in" = {bold(I)_1, bold(I)_2, ..., bold(I)_F, bold(D)_"semi", bold(K), bold(T)} $

        Where:
        - $bold(I)_f in bb(R)^(H times W times 3)$: RGB frames ($F$ frames)
        - $bold(D)_"semi" in bb(R)^(N times 3)$: Semi-dense 3D points
        - $bold(K) in bb(R)^(3 times 3)$: Camera intrinsics matrix
        - $bold(T)_f in "SE"(3)$: Camera pose for frame $f$

        #v(0.3em)
        Multiple camera streams supported:
        - RGB (high-res)
        - SLAM cameras (grayscale, rectified)
      ]
    ],
    [
      #color-block(title: [Output Formulation])[
        *3D Occupancy Volume*:
        $ bold(V)_"out" in bb(R)^(D_x times D_y times D_z times C) $

        - Voxel grid dimensions: $D_x times D_y times D_z$
        - $C$ channels for:
          + Occupancy probability
          + Object class scores
          + Surface normals

        *Detected Objects*:
        $ cal(O) = {(bold(b)_i^"3D", c_i, s_i)}_(i=1)^N $

        - $bold(b)_i^"3D" in bb(R)^9$: Oriented bounding box
        - $c_i$: Object class
        - $s_i$: Confidence score
      ]
    ],
  )

  #color-block(title: [Feature Lifting Process])[
    1. *2D Feature Extraction*: Frozen DinoV2.5 backbone
      $ bold(F)_"2D" = phi_"DINOv2.5"(bold(I)_f) in bb(R)^(H' times W' times D_"feat") $

    2. *3D Projection*: For each voxel $bold(v) in bb(R)^3$, aggregate features from all frames
      $ bold(F)_"3D"(bold(v)) = "Aggregate"({pi(bold(T)_f^(-1) bold(v), bold(K), bold(F)_"2D"^f)}_( f=1)^F) $

      where $pi(dot)$ is the camera projection function

    3. *3D Convolution*: Process lifted features
      $ bold(V)_"out" = psi_"3D-CNN"(bold(F)_"3D", bold(D)_"semi") $
  ]
]

// ATEK Section
#section-slide(
  title: [ATEK Toolkit],
  subtitle: [
    Streamlined ML Workflows for Aria Datasets
    #figure(image(fig_path + "atek/overview.png", width: 20cm), caption: [@ATEK-Repo])
  ],
)//[
// #align(center)[
//   #text(size: 14pt)[


//     #v(2em)

//     #grid(columns: (1fr, 1fr, 1fr), gutter: 2em,
//       [*Data Store*\ PyTorch\ WebDataset],
//       [*Evaluation*\ Mesh Metrics\ Benchmarks],
//       [*Training*\ Pre-processed\ Splits]
//     )
//   ]
// ]
// ]

#slide(title: [ATEK Toolkit])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5cm,
    [
      #color-block(title: [ATEK Data Store])[
        - Pre-processed for various tasks $->$ ready for PyTorch training
        - Local download or cloud streaming
        - Eval metrics (accuracy, completeness, F-score) $->$ adaptation for #RRI
        - Integration w/ Meta's MPS
        - Various example notebooks
      ]

      #color-block(title: [Provided Models])[
        - #textit[Cube R-CNN] @omni3d-cubercnn-brazil2023 for OBBs
        - #textit[EFM] @EFM3D-straub2024 for OBBs & surface reconstruction
      ]
    ],
    [
      #v(1em)
      #quote-block[
        *Resources*
        - #link("https://github.com/facebookresearch/ATEK")[ATEK GitHub] @ATEK-about-2025
        - #link(
            "https://www.youtube.com/watch?v=m6oFLfYUpoM&t=7242s",
          )[ECCV 2024 Tutorial: Egocentric Research with Project Aria]
        - Atek Context7 ID: `/facebookresearch/atek`
      ]

      #v(1em)
      #text(size: 15pt)[
        ATEK provides #emph-color[streamlined ML workflows] for rapid prototyping and benchmarking on Aria datasets.
      ]
    ],
  )
]

// TODO slide
#slide(title: [Next Steps & TODOs])[
  #color-block(title: [Literature Review])[
    - Read Project Aria paper @ProjectAria-ASE-2025
    - Study #EFM3D & #EVL architecture in depth @EFM3D-straub2024
    - Deep dive into GenNBV's multi-source embeddings @GenNBV-chen2024
    - Compare VIN-NBV vs. GenNBV: #RRI prediction vs. coverage-based rewards
  ]

  #color-block(title: [Technical Exploration])[
    - Explore GT meshes (#code-inline[.ply] files) in #ASE dataset
    - Get familiar with #link("https://github.com/facebookresearch/ATEK")[ATEK] and #link("https://github.com/facebookresearch/ATEK/blob/main/docs/ATEK_Data_Store.md")[ATEK Data Store]
    - Test mesh-based evaluation metrics (accuracy, completeness, F-score)
    - Experiment with probabilistic 3D occupancy grids
  ]

  #color-block(title: [Implementation Goals])[
    - Implement ray-casting for mesh-based visibility computation
    - Develop entity-wise #RRI computation pipeline using GT meshes
    - Design #FiveDoF action space for scene exploration
    - Build multi-source state embedding (geometric + semantic + action)
    - Prototype #RRI prediction network architecture
  ]
]

// VIN-NBV Overview
#slide(title: [VIN-NBV: Learning-Based Next-Best-View])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Key Innovation])[
        - First #NBV method to directly optimize #emph-bold[reconstruction quality] (not coverage)
        - Predicts #emph-color[Relative Reconstruction Improvement (#RRI)] without capturing new images
        - 30% improvement over coverage-based baselines
        - Trained 24h on 4 A6000 GPUs @VIN-NBV-frahm2025
      ]

      #color-block(title: [Relative Reconstruction Improvement], spacing: 0.5em)[
        For a candidate view $bold(q)$, #RRI quantifies expected improvement:

        $
          RRI(bold(q)) = (CD(cal(R)_"base", cal(R)_"GT") - CD(cal(R)_("base" union bold(q)), cal(R)_"GT")) / (CD(cal(R)_"base", cal(R)_"GT"))
        $

        - Range: $[0, 1]$ where higher = better view
        - Normalized by current error $arrow.r$ scale-independent
        - #CD measures reconstruction quality
      ]
    ],
    [
      #color-block(title: [VIN Architecture])[
        Predicts #RRI from current reconstruction state:

        $ hat(RRI)(bold(q)) = "VIN"_theta (cal(R)_"base", bold(C)_"base", bold(C)_bold(q)) $

        - *Input*: Partial point cloud + camera poses
        - *Features*: Surface normals, visibility counts, depth, coverage
        - *Output*: Predicted #RRI via ordinal classification (15 bins)
      ]

      #v(1em)
      #quote-block(color: rgb("#285f82"))[
        VIN-NBV demonstrates that #emph-it[learning reconstruction-aware NBV policies] significantly outperforms traditional coverage-based approaches.
      ]
    ],
  )
]

// GenNBV Overview
#slide(title: [GenNBV: Generalizable Next-Best-View Policy])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Key Innovations])[
        - #emph-bold[#FiveDoF free-space action space]: 3D position + 2D rotation (yaw, pitch)
        - #emph-color[Multi-source state embedding]: geometric, semantic, action representations
        - #emph-it[Probabilistic 3D occupancy grid] vs. binary (distinguishes unscanned from empty)
        - Cross-dataset generalization: 98.26% coverage on Houses3K, 97.12% on OmniObject3D
      ]

      #color-block(title: [Action Space Design])[
        $ cal(A) = underbrace(bb(R)^3, "position") times underbrace(S O(2), "heading") $

        - Approximately 20m x 20m x 10m position space
        - Omnidirectional heading subspace
        - #emph-it[No hand-crafted constraints] (e.g., hemisphere)
      ]
    ],
    [
      #color-block(title: [State Representation])[
        *Geometric Embedding* $bold(s)_t^G$:
        - Probabilistic 3D occupancy grid from depth maps
        - Bresenham ray-casting with log-odds update
        - Three states: #emph-color[occupied], #emph-color[free], #emph-color[unknown]

        *Semantic Embedding* $bold(s)_t^S$:
        - RGB images $->$ grayscale $->$ 2D CNN
        - Helps distinguish holes from incomplete scans

        *Action Embedding* $bold(s)_t^A$:
        - Historical viewpoint sequence encoding

        *Combined*: $bold(s)_t = "Linear"(bold(s)_t^G; bold(s)_t^S; bold(s)_t^A)$
      ]

      #v(0.5em)
      #quote-block[
        RL-based framework with PPO. Reward: $Delta$#CR between steps. @GenNBV-chen2024
      ]
    ],
  )
]

// Reconstruction Metrics Theory
#slide(title: [Reconstruction Quality Metrics])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [Surface-to-Surface Distance Metrics])[
        *Accuracy* (Prediction $->$ GT):
        $ "Acc" = (1)/(|cal(P)|) sum_(bold(p) in cal(P)) min_(bold(q) in cal(M)_"GT") ||bold(p) - bold(q)||_2 $

        *Completeness* (GT $->$ Prediction):
        $ "Comp" = (1)/(|cal(M)_"GT"|) sum_(bold(q) in cal(M)_"GT") min_(bold(p) in cal(P)) ||bold(p) - bold(q)||_2 $

        Where:
        - $cal(P)$: Predicted PC from dense or semi-dense reconstruction or sampled from pred mesh
        - $cal(M)_"GT"$: Sampled points from GT mesh
      ]
    ],
    [
      #color-block(title: [Precision, Recall & F-score])[
        At threshold $tau$ (typically 5cm):

        $
          "Pr"_(@tau) = (|{bold(p) in cal(P) : min_(bold(q) in cal(M)_"GT") ||bold(p) - bold(q)|| < tau}|)/(|cal(P)|)
        $

        $
          "Re"_(@tau) = (|{bold(q) in cal(M)_"GT": min_(bold(p) in cal(P)) ||bold(p) - bold(q)|| < tau}|)/(|cal(M)_"GT"|)
        $

        $ "F-score"@tau = (2 dot "Precision" dot "Recall")/("Precision" + "Recall") $
      ]

      #v(0.5em)
      #color-block(title: [Chamfer Distance (Bidirectional)])[
        $ CD(cal(P), cal(M)_"GT") = "Acc" + "Comp" $

        Combines both directions of surface error
      ]
    ],
  )

  #quote-block[
    *ATEK Implementation*:
    - `evaluate_single_mesh_pair()`#footnote[#link("https://github.com/facebookresearch/ATEK/blob/main/atek/evaluation/surface_reconstruction/surface_reconstruction_metrics.py")[src]] computes all metrics using:
      - #link("https://github.com/mikedh/trimesh")[trimesh.Trimesh]: Load meshes + sample surfaces uniformly
      - `compute_pts_to_mesh_dist()`: Point-to-mesh distance via batched triangle projection
      - `point_to_closest_tri_dist()`: Barycentric coordinate projection test + plane distance
      - Fallback: `point_to_closest_vertex_dist()` when projection fails

    See #link("../../contents/metrics.qmd")[`metrics.qmd`] for detailed formulas and algorithm explanations.
  ]
]

// RRI Computation with Meshes Slide
#slide(title: [Computing RRI with GT Meshes])[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [RRI from GT Mesh])[
        Given:
        - $cal(M)_"GT"$: GT mesh (from #ASE `.ply` files)
        - $cal(P)_t$: Current reconstruction from first $t$ views
        - $bold(q) in "SO"(2) times.l RR^3$: Candidate viewpoint, 5DoF (position + yaw, pitch)
        - $cal(P)_(t union bold(q))$: Updated reconstruction after capturing from $bold(q)$

        #v(0.5em)
        *Mesh-based RRI (oracle)*:
        $
          "RRI"(bold(q)) = frac(CD(cal(P)_t, cal(M)_"GT") - CD(cal(P)_(t union bold(q)), cal(M)_"GT"), CD(cal(P)_t, cal(M)_"GT"))
        $


      ]

      #color-block(title: [Key Functions from #EFM3D & ATEK], spacing: 0.5em)[
        #text(size: 14pt)[
          *Point Cloud Generation*:
          - `dist_im_to_point_cloud_im()`: Depth $->$ 3D points
          - `collapse_pointcloud_time()`: Merge temporal PCs
          - `pointcloud_to_voxel_counts()`: PC $->$ density grid

          *Ray-Mesh Operations*:
          - `ray_obb_intersection()`: Ray-box intersection
          - `sample_depths_in_grid()`: Sample depths along rays

          *Distance Computation*:
          - `compute_pts_to_mesh_dist()`: Min distance to triangles
          - `eval_mesh_to_mesh()`: Full evaluation pipeline
        ]
      ]
    ],
    [
      #color-block(title: [RRI Oracle Pipeline])[
        1. Load GT mesh from #ASE

        2. Build $cal(P)_t$ from captured views
          - dense PC from depth maps
          - or semi-dense SLAM PC#footnote[`semidense_points.csv`]

        3. Simulate view from $bold(q)$
          - Ray-cast to $cal(M)_"GT" arrow.r cal(P)_bold(q)$

        4. Merge: $cal(P)_(t union bold(q)) = cal(P)_t union cal(P)_bold(q)$
          - Voxel downsample for consistency (e.g., 1cm)

        5. Compute #RRI using CD metric
      ]
    ],
  )
]


// Putting it together: RRI & NBV with ASE, EFM3D & ATEK
#slide(title: [RRI-based NBV for Scene-Level Reconstruction])[
  #color-block(title: [VIN with EVL Backbone])[
    // *From VIN-NBV*: Direct reconstruction quality optimization via #RRI

    // *From GenNBV*: Free-space exploration + multi-source state embedding
    // Here elaborate on how we could use the visibility information.
    // What Input features to the VIN Network
    // EVL backbone to encode all observations until t = tau, what latent features to feed into the VIN head?

    *Our Approach*: Adapt #RRI prediction to #emph-color[scene-level] environments with #FiveDoF action space
  ]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    [
      #color-block(title: [#RRI with GT Meshes], spacing: 0.4em)[
        Use #ASE visibility data + GT meshes for #emph-it[oracle #RRI]:

        $
          RRI(bold(q)) = (d(cal(P)_"partial", cal(M)_"GT") - d(cal(P)_"partial" union bold(q), cal(M)_"GT")) / (d(cal(P)_"partial", cal(M)_"GT"))
        $

        where $cal(M)$ represents meshes, $d(dot, dot)$ is mesh distance
      ]
    ],
    [
      #color-block(title: [Proposed Pipeline])[
        1. *Reconstruct*: Build $cal(P)_"partial"$ from historical trajectory
        2. *Sample*: Generate candidate viewpoints in free space around latest pose
        3. *Compute Features*: Extract geometric + semantic embeddings from *EVL*
        4. *Predict*: Use *VIN* to predict #RRI per candidate
        5. *Select*: Choose #NBV based on #RRI
      ]
    ],
  )

  #quote-block(color: rgb("#fc5555"))[
    *Key Challenge*: Ray-casting from candidate views to compute visibility on GT meshes for entity-wise #RRI computation
  ]

  #quote-block(color: rgb("#fc5555"))[
    *Extension Entity-wise #RRI*:

    $ RRI_"total" = sum_(e in cal(E)) w_e dot RRI_e #h(1fr) "where" cal(E) = "{walls, doors, objects, ...}" $
  ]
  - This could be done by segmenting the GT meshes and PCs per entity type and computing the RRI separately.
]



// Bibliography slide
#slide(title: [Bibliography])[
  #bibliography("/references.bib", style: "/ieee.csl")
]

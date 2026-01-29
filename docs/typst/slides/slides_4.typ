// Final presentation slides for Aria-NBV (Oracle RRI + VIN v3 baseline).
//
// Target audience: my Professor whose expertise is 3D Deep Learning in Visual Computing, he is already familiar with the project from various update meetings and our previous slides_{1,2,3}.typ, so we can skip basic explanations of concepts;
// Goal: present the final architecture, empirical findings, and limitations to guide next steps
// for the master-thesis continuation (dataset scale + compute needs).
//
// NOTE: numeric values must be imported from data artifacts (cache metadata, W&B summaries,
// Optuna exports) to avoid inconsistencies.
// Target structure:
// Overview of implemented components
// Dataset + offline dataset (coverage, important distributions, batching, data contracts)
// Oracle RRI pipeline (candidates, rendering, backprojection, scoring (P <-> M metrics, RRI metric))
// VIN v3 architecture
//  - general architecture
//  - feature branches + I/O forumlations (frames + shapes + computation flow within each branch)
//  - CORAL head + losses + binning
// - training dynamics + metrics logged
// - evidence so far (Optuna patterns + best run)
// - limitations + next steps

#import "template.typ": *
#import "notes.typ": *
#import "@preview/muchpdf:0.1.1": muchpdf
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

// Shared macros and symbols (paper + slides)
#import "../shared/macros.typ": *

#let fig_path = "../../figures/"

// ---------------------------------------------------------------------------
// Imported slide data (single source of truth for numbers shown in the deck)
// ---------------------------------------------------------------------------

#let cache = json("/typst/slides/data/offline_cache_stats.json").offline_cache
#let wb = json("/typst/slides/data/wandb_rtjvfyyp_summary.json").wandb
#let wb_dyn = json("/typst/slides/data/wandb_rtjvfyyp_dynamics.json").wandb_dynamics
#let v3_vs_t41 = json("/typst/slides/data/vin_v3_01_vs_t41_summary.json").vin_v3_01_vs_t41
#let top_trials = csv("/typst/paper/data/optuna_v2_top_trials.csv", row-type: dictionary)
#let labeler_cfg = toml("/typst/slides/data/paper_figures_oracle_labeler.toml")
#let gen_cfg = labeler_cfg.labeler.generator
#let depth_cfg = labeler_cfg.labeler.depth
#let renderer_cfg = depth_cfg.renderer

#let round(x, digits: 3) = {
  let s = calc.pow(10, digits)
  calc.round(x * s) / s
}

#let pct(x, digits: 1) = round(100 * x, digits: digits)

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Munich University of Applied Sciences],
  logo: [#image(fig_path + "hm-logo.svg", width: 2cm)],
  config-info(
    title: [Aria-NBV: Oracle RRI + VIN v3 Candidate Scoring],
    subtitle: [Offline oracle supervision, diagnostics, and learned NBV baseline],
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
  config-colors(
    primary: theme_color_primary_hm,
    lite: theme_color_block,
  ),
)

// Global style overrides
#set text(size: 17pt, font: "Open Sans")
#show figure.caption: set text(size: 12pt, weight: "medium", fill: theme_color_footer.darken(40%))
#show grid: set grid(columns: (1fr, 1fr), gutter: 0.8cm)
#show cite: set text(size: 10pt)
#show bibliography: set text(size: 14pt)
#show link: set text(fill: blue)
#show link: it => underline(it)

// ---------------------------------------------------------------------------
// Title + agenda
// ---------------------------------------------------------------------------

#title-slide()

// #slide(title: [Agenda])[
//   #color-block(title: [What we cover])[
//     - Oracle RRI pipeline: candidates, rendering, backprojection, point-to-mesh scoring.
//     - Offline dataset + batching contract (cache, indices, `VinOracleBatch`, `VinSnippetView`).
//     - VIN v3 architecture: EVL voxel context + per-candidate evidence + CORAL head.
//     - Objective + metrics: what we log and why (diagnostics-first).
//     - Evidence so far: Optuna sweep patterns + best W&B run (#code-inline[#wb.run_id]).
//     - Limitations + master-thesis next steps (dataset scale, compute, stability).
//   ]
//   #good-note(width: 100%)[
//     All numeric values are imported from artifacts under `docs/typst/slides/data/` and the offline cache metadata.
//   ]
// ]

// // ---------------------------------------------------------------------------
// // Motivation + scope
// // ---------------------------------------------------------------------------

// #section-slide(
//   title: [Motivation and Scope],
//   subtitle: [From expensive oracle supervision to a lightweight candidate scorer],
// )

// #slide(title: [Why oracle RRI?])[
//   #grid(
//     [
//       #color-block(title: [Motivation])[
//         - Proxy objectives (coverage, entropy, novelty) can miss occlusion and surface-detail effects.
//         - Oracle RRI provides direct reconstruction-quality supervision for candidate ranking.
//         - We treat oracle correctness as a first-class deliverable (diagnostics-first).
//       ]
//       #color-block(title: [Scope])[
//         - Discrete candidate set around a reference rig pose.
//         - Offline depth rendering + point-to-mesh evaluation to compute RRI labels.
//         - Learned VIN v3 scorer predicts ordinal RRI from EVL + per-candidate evidence.
//       ]
//     ],
//     [
//       #figure(
//         image(fig_path + "VIN-NBV_diagram.png", width: 100%),
//         caption: [VIN-NBV reference pipeline. We retain view-conditioned projection bias while adapting to EVL. @VIN-NBV-frahm2025],
//       )
//     ],
//   )
// ]

// #slide(title: [What is implemented (today)])[
//   #grid(
//     [
//       #color-block(title: [Oracle label pipeline])[
//         - Candidate view generation with collision / free-space checks.
//         - Mesh depth rendering (PyTorch3D) and metric backprojection.
//         - Point-to-mesh evaluation (accuracy + completeness) #sym.arrow.r per-candidate RRI.
//       ]
//       #color-block(title: [Offline data product])[
//         - Offline cache with indices and train/val split.
//         - Typed batching contract: `VinOracleBatch` + padded `VinSnippetView`.
//         - Candidate shuffling toggle (TODO) to avoid ordering bias.
//       ]
//     ],
//     [
//       #color-block(title: [VIN v3 baseline scorer])[
//         - EVL voxel field #sym.arrow.r pose-conditioned global context (#symb.vin.global).
//         - Per-candidate semidense projection stats + small grid CNN.
//         - Optional trajectory encoder (enabled in best run).
//         - CORAL ordinal head and ranking diagnostics.
//       ]
//       #quote-block[
//         Thesis direction: scale oracle-labeled data and stabilize training dynamics to unlock stronger learned NBV policies.
//       ]
//     ],
//   )
// ]

// ---------------------------------------------------------------------------
// Data + oracle pipeline
// ---------------------------------------------------------------------------

#section-slide(
  title: [Data and Oracle Pipeline],
  subtitle: [
    Candidate generation + depth rendering + RRI computation

    #figure(
      image(fig_path + "app-paper/data_frames_81022_11.png", width: 100%),
      caption: [First/last RGB, SLAM-L/R, and depth frames (example snippet).],
    )

  ],
)

#slide(title: [ASE ATEK Dataset])[
  #grid(
    [
      #color-block(title: [ASE ATEK overview (what we need)], spacing: 0.5em)[
        #set text(size: 15pt)
        - Scale (full ASE): 100k scenes, 58M+ RGB frames, 67 days.
        - Per snippet (ATEK/EFM view):
          + Trajectory #symb.ase.traj (20 frames \@10 Hz; 2 s window).
          + Semi-dense SLAM points #symb.ase.points_semi.
          + 3M OBBs (43 classes).
        - Local shard snapshot (`.data/ase_efm`): 100 scenes, 576 shards, 4,608 snippets (~49 GB).
      ]
    ],
    [

      #color-block(title: [ATEK-EFM snippet format (key facts)], spacing: 0.5em)[
        #set text(size: 15pt)
        - WebDataset shards: per-scene folders with `shards-*.tar` (streamed, then windowed).
        - Window stride: 10 frames (1 s overlap between consecutive 2 s snippets).
        - Resized streams: RGB 240x240, SLAM ~320x240 (calibration + rig poses preserved).
        - GT meshes: #symb.ase.mesh for 100 validation scenes (watertight).
      ]
    ],
  )
  #grid(
    columns: (1fr,),
    [
      #figure(
        image(fig_path + "app-paper/scene_view_81022_11.png", width: 92%),
        caption: [One snippet: #symb.ase.mesh + #symb.ase.points_semi + #symb.ase.traj + camera frustum.],
      )
    ],
  )
]

// #slide(title: [Mesh subset + cached snippet distribution])[
//   #grid(
//     [
//       #figure(
//         image(fig_path + "gt_mesh_manhattan_sample.png", width: 100%),
//         caption: [GT mesh example (oracle supervision target).],
//       )
//     ],
//     [
//       #figure(
//         image(fig_path + "ase_efm_snippet_hist.png", width: 100%),
//         caption: [Snippet counts per scene in the current local snapshot.],
//       )
//     ],
//   )
// ]

#slide(title: [Oracle RRI Pipeline])[

  #figure(
    muchpdf(read(fig_path + "diagrams/oracle_rri/oracle_rri_compact.pdf", encoding: none), width: 100%),
    caption: [Oracle RRI Pipeline.],
  )
  #grid(
    columns: (1fr, 1fr),
    [
      #color-block(title: [Stages])[
        1. Sample candidate poses #symb.oracle.candidates around the reference rig pose.
        2. Render candidate depth maps #symb.oracle.depth_q (metric z-buffer from #(symb.ase.mesh)).
        3. Backproject valid depths #sym.arrow.r candidate point clouds #symb.oracle.points_q.
        4. Evaluate point-to-mesh quality before/after and compute RRI.
      ]
      // #color-block(title: [Reusable compute path])[
      //   - `OracleRriLabeler` orchestrates generation #sym.arrow.r rendering #sym.arrow.r scoring.
      //   - Same components are used in Streamlit diagnostics and offline preprocessing.
      //   - Output is a single batch with all intermediate tensors needed for debugging.
      // ]
    ],
    [
      #figure(
        image(fig_path + "impl/oracle-rri-sample.png", width: 85%),
        caption: [Oracle RRI Sample.],
      )
    ],
  )
]

//  Now slides on the different stages, on each of these slides include a table with the key parameters, and include a place holder figure for relevant visualizations!
#slide(title: [Candidate Generation])[
  #grid(
    columns: (2fr, 1fr),
    [
      #io-formulation(
        [
          - Trajectory #symb.ase.traj
          - GT Mesh #symb.ase.mesh
        ],
        [
          - Candidate views $#symb.oracle.candidates subset "SE"(3)^#symb.shape.Nq$ + cameras #symb.oracle.cameras_q
          - Reference pose #symb.ase.traj_final
          - Valid + debug masks
        ],
      )
      #color-block(title: [Summary])[
        - Sample candidate centers on a constrained shell around the reference pose.
        - Assign view directions (radial-away or forward) and optional jitter.
        - Prune using collision + free-space rules and min-distance checks.
      ]
    ],
    [
      #figure(
        kind: "table",
        supplement: [Table],
        caption: [Key parameters (candidate generation).],
        text(size: 12pt)[
          #let gen_rows = (
            ([$#symb.shape.Nq$], [#gen_cfg.num_samples]),
            ([$r_"min"$], [#gen_cfg.min_radius]),
            ([$r_"max"$], [#gen_cfg.max_radius]),
            ([$theta_"min"$], [#gen_cfg.min_elev_deg]),
            ([$theta_"max"$], [#gen_cfg.max_elev_deg]),
            ([$psi_"span"$], [#gen_cfg.delta_azimuth_deg]),
            ([$psi_delta$], [#gen_cfg.view_max_azimuth_deg]),
            ([$theta_delta$], [#gen_cfg.view_max_elevation_deg]),
            ([$phi_delta$], [#gen_cfg.view_roll_jitter_deg]),
            (code-inline("align_to_gravity"), [#gen_cfg.align_to_gravity]),
            (code-inline("min_dist_to_mesh"), [#gen_cfg.min_distance_to_mesh]),
          )
          #let gen_cells = gen_rows.flatten()
          #table(
            columns: (14em, auto),
            align: (left, left),
            toprule(),
            table.header([Setting], [Value]),
            midrule(),
            ..gen_cells,
            bottomrule(),
          )
        ],
      )
    ],
  )
]

#slide(title: [Candidate Generation: Position sampling])[
  #grid(
    columns: (1.2fr, 1fr),
    [
      #color-block(title: [Position sampling])[
        - Sample directions on the unit sphere $bb(S)^2 subset bb(R)^3$, then rescale into $(psi, theta)$ caps.
        - Gravity-aligned sampling uses #T(symb.frame.w, symb.frame.s) (roll/pitch removed).
        - #T(symb.frame.w, symb.frame.r) is the reference pose rotation; $(psi, theta)$ only
          parameterize the direction $bold(s)_q in bb(S)^2$.

        #v(0.3em)
        $
          bold(s)_q ~ cal(U)(bb(S)^2),
          quad
          r_q ~ cal(U)(r_"min", r_"max")
        $
        $
          (psi, theta) = "angles"(bold(s)_q) \
          (psi, theta) <- "lin"([psi_"min", psi_"max"] times [theta_"min", theta_"max"]; (psi, theta))
        $
        $
          bold(s)_q' = (cos theta sin psi, sin theta, cos theta cos psi)
        $
        $
          #(symb.oracle.center) _q = #T(symb.frame.w, symb.frame.r) (r_q bold(s)_q')
        $
        #cite(<Formelsammlung-papula2024>, supplement: [p.43])
      ]
    ],
    [
      #figure(
        image(fig_path + "app-paper/pos_ref.png", width: 100%),
        caption: [Candidate centers in the reference frame.],
      )
    ],
  )
]

#slide(title: [Candidate Generation: View directions])[
  #grid(
    columns: (1.2fr, 1fr),
    [
      #color-block(title: [Direction + pose])[
        - Base orientation from `view_direction_mode` (e.g., radial-away).
        - $(psi, theta)$ caps apply to the *jitter delta* (box-uniform), not the base view.
        - Compose candidate pose #T(symb.frame.w, symb.frame.cq) from center + base + delta.

        #v(0.3em)
        $
          bold(R)_("base") = "look-away"(#symb.frame.cq, #(symb.ase.traj_final)) \
          psi ~ cal(U)(-psi_delta/2, psi_delta/2),
          theta ~ cal(U)(-theta_delta/2, theta_delta/2)
        $
        $
          #T(symb.frame.w, symb.frame.cq) =
          (bold(R)_("base") compose bold(R)_("delta"), #symb.frame.cq)
        $
      ]
    ],
    [
      #figure(
        image(fig_path + "app-paper/view_dirs_ref.png", width: 100%),
        caption: [View direction density (azimuth/elevation).],
      )
    ],
  )
]

#slide(title: [Candidate Generation: Jitter + rules])[
  #grid(
    columns: (1fr, 1.6fr),
    [
      #color-block(title: [View jitter])[
        - Box-uniform jitter in yaw/pitch within caps:
          $
            psi ~ cal(U)(-psi_delta /2, psi_delta/2) \
            theta ~ cal(U)(-theta_delta/2, theta_delta/2).
          $
        - Optional roll:
        $
          phi ~ cal(U)(-phi_delta, phi_delta)
        $

        $
          bold(R)_(delta) = bold(R)_z (psi) bold(R)_y (theta) bold(R)_x (phi) \
          #T(symb.frame.w, symb.frame.cq) = #T(symb.frame.w, symb.frame.cq) compose mat(bold(R)_(delta), bold(0); bold(0)^T, 1)
        $
      ]
    ],
    [
      #figure(
        image(fig_path + "app-paper/orientation_jitter.png", width: 105%),
        caption: [Orientation jitter distribution (delta yaw/pitch/roll, deg).],
      )
      #figure(
        image(fig_path + "app-paper/ypr_reference.png", width: 105%),
        caption: [Reference-frame yaw/pitch/roll distribution (deg).],
      )
      #v(0.3em)
      #color-block(title: [Pruning Rules])[
        - *Rules*: min distance to mesh, collision-free ray, free-space AABB.
      ]
    ],
  )
]
//  TODO: currently we just replace invalid candidates and mask them if Nq is not reached


#slide(title: [Candidate Depth Rendering])[
  #grid(
    columns: (1.35fr, 1fr),
    gutter: 0.4cm,
    [
      #text(size: 17pt)[
        #io-formulation(
          [
            - Candidate views #symb.oracle.candidates, #symb.oracle.cameras_q
            - GT mesh #symb.ase.mesh, #symb.ase.faces
          ],
          [
            - Depth maps $#symb.oracle.depth_q$
            - Valid mask $bold(M)_q$
            - #symb.oracle.cameras_q' (P3D cameras)
          ],
        )
      ]
      #v(0.3em)
      #color-block(title: [Transforms + ops])[
        - Build PyTorch3D #symb.oracle.cameras_q' with extrinsics #T(symb.frame.w, symb.frame.cq) and $(#symb.shape.Wdim / 2, #symb.shape.H / 2)$ from #code-inline("CameraTW").
        - Rasterize #symb.ase.mesh to depth $#symb.oracle.depth_q subset R^(#symb.shape.Nq times #symb.shape.H' times #symb.shape.Wdim')$.
        - Valid mask: #text(size: 15pt)[#code-inline("pix_to_face") $>=$ 0 and #code-inline("znear") $<$ #symb.oracle.depth_q $<$ #code-inline("zfar")]\
          - $bold(M)_q subset {0,1}^(#symb.shape.Nq' times #symb.shape.H' times #symb.shape.Wdim)$
      ]
    ],
    [

      #figure(
        image(fig_path + "app-paper/cand_renders_1x3.png", width: 100%),
        caption: [Candidate depth renders (1x3).],
      )
      #figure(
        image(fig_path + "app-paper/depth_histograms_3x3.png", width: 100%),
        caption: [Depth histograms across candidates (3x3).],
      )
    ],
  )
]

#slide(title: [Backprojection])[
  #grid(
    columns: (1.15fr, 1fr),
    gutter: 0.4cm,
    [
      #text(size: 16pt)[
        #io-formulation(
          [
            - Depth maps #symb.oracle.depth_q + #symb.oracle.cameras_q
            - Semi-dense PC $#(symb.ase.points_semi)$
          ],
          [
            - Candidate PCs $#symb.oracle.points_q subset R^(#symb.shape.Nq times #symb.shape.P times 3)$
            - $ell_q subset {0, dots, #symb.shape.H dot #symb.shape.Wdim}^(#symb.shape.Nq)$
            - AABBs $bold(b)_{"aabb"} subset R^6$
          ],
        )
      ]
    ],
    [
      #figure(
        image(fig_path + "app-paper/backproj+semi.png", width: 60%),
        caption: [Backprojected candidate points + semi-dense SLAM points.],
      )
    ],
  )
  #color-block(title: [Transforms + ops])[
    + Subsample pixel centers $(u, v)$ by stride $k$.
    + Map to NDC with $s = min(#symb.shape.H, #symb.shape.Wdim)$ @PyTorch3D-Cameras-2025:
      $
        x_"ndc" = - (u + 1/2 - #symb.shape.Wdim'/2) (2/s) quad
        y_"ndc" = - (v + 1/2 - #symb.shape.H'/2) (2/s)
      $
    + Unproject: $bold(p)_"world" = "unproject"(x_"ndc", y_"ndc", d_q)$, where $d_q$ is sampled from #symb.oracle.depth_q.
    + Pad per-candidate PCs; fuse with $#(symb.ase.points_semi)$ for AABB cropping.
  ]
]



#slide(title: [Oracle RRI: Accuracy + Completeness])[
  #grid(
    columns: (1.1fr, 1fr),
    [
      #good-note[$ #eqs.rri.acc $]
      #figure(
        image(fig_path + "app-paper/acc_top10.png", height: 50%),
        caption: [Candidates by accuracy (lower is better).],
      )
    ],
    [
      #good-note[$ #eqs.rri.comp $]
      #figure(
        image(fig_path + "app-paper/comp_top10.png", height: 50%),
        caption: [Candidate completeness (lower is better).],
      )
    ],
  )

  - Crop #symb.ase.mesh to AABB, then
    compute $bold(cal(P))_bullet <-> #symb.ase.mesh$ distances with PyTorch3D.
  - $cal(A)$ dominated by #symb.ase.points_semi #sym.arrow not discriminative w.r.t. candidates

]

#slide(title: [Oracle RRI: Relative improvement])[
  #figure(
    image(fig_path + "app-paper/oracle_rri_bar.png", width: 100%),
    caption: [Per-candidate oracle RRI (bar chart).],
  )
  #align(center)[
    #good-note[$ #eqs.rri.rri, quad #eqs.rri.union $]
  ]
]


#slide(title: [Oracle RRI distribution])[
  #grid(
    [
      #color-block(title: [Skewed candidate gains])[
        - Most candidates yield marginal improvements.
        - A small fraction produce large RRI gains.
        - Diagnostics log accuracy and completeness terms to catch failure cases early.
      ]
    ],
    [
      #figure(
        image(fig_path + "app/rri_hist_81056_000022.png", width: 100%),
        caption: [Oracle RRI histogram for one example snippet.],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// Offline cache + batching
// ---------------------------------------------------------------------------

#section-slide(
  title: [Offline Dataset and Batching],
  subtitle: [Fast training iterations without recomputing the oracle],
)

//  TODO: motivate usage of offline dataset:
// OracleRRI Pipeline uses pytorch - cannot be parallelized; pipeline run takes approx 30s per snippet for #symb.shape.Nq = 60 candidates
// Should cache backbone outputs once (for each of the 4608 snippets) as single forward requires 8+ GB GPU memory and takes up to 60s
// Training signal is very noisy - higher batch size stabilizes gradients
// Now we can easily train with B >= 16; single epoch on 798 samples takes approx 8 minutes vs > 24 hours if we compute oracle and EVL scene encoding on-line.
#slide(title: [Offline cache: Motivation])[
  #color-block(title: [Why offline?])[
    - Oracle pipeline uses P3D rendering + backprojection mesh ops.
      + >30s for #symb.shape.Nq = 60
      + cannot be parallelized
    - Single EVL forward-pass requires:
      + 8+ GB VRAM
      + >60s per sample
    - Offline cache enables:
      + batching ($nabla$ stablization)
      + reproducible splits & validation monitoring
      + >180x speedup (8 min vs >24 h)
      + enables HParam sweeps
  ]
]

#slide(title: [Offline cache: coverage + footprint])[
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 0.35cm,
    [
      #color-block(title: [Coverage numbers])[
        - Cached scenes: #cache.unique_scenes / #cache.meta_scenes_total (#pct(cache.meta_scenes_covered_frac)%)
        - Cached samples: #cache.index_entries / #cache.total_snippets (_#pct(cache.index_entries / cache.total_snippets)%_)
        - Split: (train: #cache.train_entries, val: #cache.val_entries)
      ]
      #color-block(title: [Storage snapshot])[
        - EFM ATEK: 49 GB
        - Cached samples: #cache.samples_size_gb GB (current subset).
        - VinSnippet cache: #cache.vin_snippet_cache_gb GB.
        - Full coverage estimate:
          #(round(cache.full_coverage_total_gb / 1000, digits: 2)) TB
      ]
      #quote-block[Memory footprint dominated by voxel grid features in EVL backbone.]
    ],
    [
      #figure(
        image(fig_path + "offline_cache/coverage.png", width: 72%),
        caption: [Coverage snapshot for the cached subset.],
      )
      #figure(
        image(fig_path + "offline_cache/footprint.png", width: 75%),
        caption: [Per-sample memory footprint (mean).],
      )
    ],
  )
]

// #slide(title: [Offline cache: Implementation])[
//   #figure(
//     image(fig_path + "diagrams/vin_nbv/mermaid/offline_cache_training.png", width: 55%),
//     caption: [Per-sample memory footprint (mean).],
//   )
// ]

#slide(title: [Offline cache: what is stored?])[
  #grid(
    [
      #color-block(title: [Cache structure])[

        - Full Oracle pipeline outputs:
          + Candidates views #symb.oracle.candidates + #symb.oracle.cameras_q
          + Depth renders #symb.oracle.depth_q + valid mask
          + Candidate PCs #symb.oracle.points_q
          + RRI Labels: #symb.oracle.rri, #symb.oracle.acc, #symb.oracle.comp
          + EVL backbone outputs
          + VinSnippetView: #symb.ase.traj + #symb.ase.points_semi
      ]
      #quote-block[Training only requires RRI labels + #symb.oracle.candidates + #symb.oracle.cameras_q + #symb.oracle.points_q + #symb.ase.points_semi & EVL head features.]
    ],
    [
      #figure(
        caption: [Vin Oracle Cache Sample: sample + candidates + depths + points + RRI.],
      )[
        #image(fig_path + "impl/vin-oracle-cache-sample.png", width: 100%)
      ]
    ],
  )
]
#slide(title: [VinOracleBatch + VinSnippetView])[
  #grid(
    color-block(title: [Key typed tensors (padded + batched)])[
      - Candidate poses: #code-inline[PoseTW[#(symb.shape.B), #(symb.shape.Nq), 12]].
      - Reference pose: #code-inline[PoseTW[#(symb.shape.B), 12]].
      - Labels: #code-inline[rri[#(symb.shape.B), #(symb.shape.Nq)]] + (#(symb.oracle.acc), #(symb.oracle.comp)) + lenghts.
      - #code-inline[PerspectiveCameras[#(symb.shape.B), #(symb.shape.Nq)]].
      - VinSnippetView: #code-inline[points_world[#(symb.shape.B), #(symb.shape.P), 3 + #(symb.shape.Csem)]] + #code-inline[lengths[#(symb.shape.B)]].
      - Per-sample candidate shuffling to avoid ordering bias.
    ],
    [
      #figure(caption: [VinOracleBatch Sample.])[
        #image(fig_path + "offline_cache/vin_oracle_batch.png", width: 100%)
      ]
    ],
  )
]


#slide(title: [Data Flow: VinDataModule + VinSnippetCache])[
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 0.35cm,
    [
      #color-block(title: [Pipeline (offline cache)], spacing: 0.5em)[
        #set text(size: 15pt)
        - #code-inline[VinDataModule] builds #code-inline[`OracleRriCacheVinDataset`].
        - #code-inline[OracleRriCacheDataset] reads #code-inline[`samples/*.pt`] and decodes depths/rri (and optional backbone) into a #code-inline[`VinOracleBatch`].
        - #code-inline[VinSnippetProviderChain] attaches a #code-inline[`VinSnippetView`]:
          #code-inline[`VinSnippetCacheProvider`] #sym.arrow.r cache hit, else #code-inline[`EfmSnippetLoader`] (#code-inline[`AseEfmDataset`]) fallback.
        - #code-inline[vin_snippet_cache_mode]: auto / required / disabled (#sym.arrow.r cache vs fallback policy).
        - #code-inline[include_efm_snippet]: attach snippet views (#code-inline[`VinSnippetView`]) vs cache-only batches.
        - #code-inline[load_backbone], #code-inline[load_depths], #code-inline[load_candidate_pcs]: trade off I/O size vs recomputation.
      ]
    ],
    [

      #figure(
        image(fig_path + "diagrams/vin_nbv/mermaid/offline_cache_training.png", width: 105%),
        caption: [Data Flow: OfflineChache #sym.arrow.r VinOracleBatch.],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// CORAL + ordinal binning (target discretization + loss)
// ---------------------------------------------------------------------------

#section-slide(
  title: [CORAL & Ordinal Binning],
  subtitle: [Skewed oracle RRI #sym.arrow.r quantile bins #sym.arrow.r ranking-aware loss + diagnostics],
)

#slide(title: [Why binning? (oracle RRI is heavy-tailed)])[
  #grid(
    [
      // Keep this slide compact: move baselines inline and slightly reduce text size.
      #set text(size: 15pt)
      #color-block(title: [Motivation (binning + CORAL)])[
        - Oracle RRI is heavy-tailed / right-skewed (many near-zero improvements, few large gains).
        - Direct regression is sensitive to outliers and stage effects (early steps tend to have larger gains). @VIN-NBV-frahm2025
        - We discretize RRI into $K=15$ *ordered* quantile bins and train an ordinal head.
        - CORAL: $K-1$ binary thresholds with shared weights, per-threshold biases; far mis-ranks flip many thresholds. @CORAL-cao2019
        - Baselines: $cal(L)_("random") approx (K-1) dot "log"(2) = 9.70$, $hat(r)_("unif") approx 0.10$.
      ]
    ],
    [
      #grid(
        columns: 1fr,
        rows: (auto, 1fr),
        gutter: 0.25cm,
        [
          #figure(
            image(fig_path + "coral/rri_distribution_linear.png", width: 100%),
            caption: [Raw oracle RRI distribution (linear counts).],
          )
        ],
      )
    ],
  )
]

#slide(title: [Quantile binning (equal-mass ordinal classes)])[
  #grid(
    [
      #color-block(title: [Binner in our code])[
        - Fit empirical quantiles on oracle RRIs (equal-mass bins):
          #block[#align(center)[#eqs.binning.edges]]
        - Assign ordinal label via edge counting (`torch.bucketize`):
          #block[#align(center)[#eqs.binning.label]]
        - This yields non-uniform bin widths (dense near $0$), but near-uniform class counts.
      ]
    ],
    [
      #figure(
        image(fig_path + "coral/rri_distribution_log_with_bin_edges.png", width: 100%),
        caption: [Log-count RRI distribution with fitted quantile edges (vertical lines).],
      )
    ],
  )
]

#slide(title: [Ordinal labels + per-bin statistics (fit data)])[
  #grid(
    [
      #color-block(title: [From labels to CORAL levels])[
        - CORAL converts each label into $K-1$ binary level targets:
          #block[#align(center)[#eqs.binning.levels]]
        - Benefits:
          + penalizes far mis-rankings more than near ones,
          + supports monotonicity diagnostics on $p_k = P(y > k)$.
      ]
    ],
    [
      #figure(
        image(fig_path + "coral/ordinal_label_histogram_fit.png", width: 100%),
        caption: [Label histogram after fitting K=15 quantile bins.],
      )
    ],
  )
]

#slide(title: [Bin calibration: midpoints, means, and variance])[
  #grid(
    [
      #color-block(title: [How we recover a scalar])[
        - CORAL predicts cumulative probabilities $p_k = P(y > k)$.
        - Convert to class marginals $pi_k$ and compute expectation:
          #block[#align(center)[#eqs.coral.marginals]]
          #block[#align(center)[#eqs.coral.expected]]
        - We initialize $u_k$ from fitted *bin means* (monotone parameterization) and monitor calibration.
      ]
    ],
    [
      #grid(
        columns: (1fr, 1fr),
        rows: auto,
        gutter: 0.35cm,
        [
          #figure(
            image(fig_path + "coral/bin_means_vs_midpoints.png", width: 100%),
            caption: [Bin means (+/- 1 std) vs midpoints (quantile bins are uneven width).],
          )
        ],
        [
          #figure(
            image(fig_path + "coral/bin_stds_vs_uniform_baseline.png", width: 100%),
            caption: [Per-bin std vs uniform baseline (width/12).],
          )
        ],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// VIN v3 architecture
// ---------------------------------------------------------------------------

#section-slide(
  title: [VIN v3 Scoring Architecture],
  subtitle: [EVL voxel context + per-candidate evidence #sym.arrow.r ordinal RRI],
)

#slide(title: [Design intent])[
  #grid(
    [
      #color-block(title: [Baseline contract])[
        - Keep the scorer small and explicit: no large point encoders (e.g., PointNeXt).
        - Prefer view-conditioned evidence that we can diagnose:
          + voxel coverage proxies,
          + semidense projection validity and moments,
          + (optional) trajectory context.
      ]
      #color-block(title: [Why view-conditioned evidence?])[
        - Ranking must depend on candidate viewpoint, not only the current scene state.
        - Pure global context can produce weak candidate separation and collapse.
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_rich_summary.png", width: 100%),
        caption: [VIN v3 module summary (implementation snapshot).],
      )
    ],
  )
]

#slide(title: [VIN-NBV (Frahm 2025) vs our VIN v3])[
  #grid(
    [
      #color-block(title: [VIN-NBV feature construction])[
        - Enrich a reconstructed point cloud with normals, visibility count, and depth.
        - Per candidate view:
          + project to a dense screen-space grid #code-inline[512x512x5] (then downsample to #code-inline[256x256]),
          + compute per-pixel variance plus pooled grid features,
          + compute an "emptiness" feature from empty pixels (inside/outside the hull).
        - Add stage cue: number of base views.
        - CNN encodes the grid to a global view feature; an MLP scores candidates.
        - Candidate visibility enters explicitly (visibility count + empty-pixel counting). @VIN-NBV-frahm2025
      ]
    ],
    [
      #color-block(title: [VIN v3 (oracle_rri) design])[
        - Use EVL voxel evidence as scene context (frozen backbone) plus semidense SLAM points.
        - Per candidate view:
          + rig-relative pose encoding (R6D + LFF),
          + semidense projection stats (visibility, coverage, depth moments; reliability-weighted),
          + coarse grid + tiny CNN for view-plane structure (#code-inline[G_sem] from config).
        - Pose-conditioned global attention pools the voxel field to a per-candidate feature vector #symb.vin.global.
        - CORAL ordinal head (fixed bins) instead of continuous regression.
        - Candidate visibility enters as a feature; training can additionally reweight losses by visibility/coverage proxies.
      ]
    ],
  )
]

#slide(title: [VIN v3: input features])[
  #grid(
    [
      #color-block(title: [Scene field (EVL)])[
        - Build #code-inline[field_in] by selecting channels from:
          occ_pr, occ_input, cent_pr_nms, counts_norm, observed, unknown, free_input, new_surface_prior.
        - Channel selection is configured via #code-inline[scene_field_channels] (not hard-coded).
        - Project to field_dim=#wb.vin_effective.field_dim channels with Conv3d+GN+GELU.
      ]
      #color-block(title: [Per-candidate cues])[
        - Pose encoding: R6D + LFF in reference rig frame.
        - Semidense projections:
          + stats (visibility, coverage, depth moments),
          + grid CNN (G=#wb.vin_effective.semidense_proj_grid_size).
        - Optional trajectory encoder: #wb.vin_effective.use_traj_encoder (enabled in best run).
      ]
    ],
    [
      #figure(
        image(fig_path + "efm3d/evl_output_summary.png", width: 100%),
        caption: [EVL output summary used to build the scene field. @EFM3D-straub2024],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// EVL backbone features (what VIN v3 consumes)
// ---------------------------------------------------------------------------

#slide(title: [EVL backbone: voxel grid contract])[
  #grid(
    [
      #color-block(title: [Voxel grid contract (frame + geometry)])[
        - Grid pose: #code-inline[t_world_voxel: PoseTW[#(symb.shape.B), 12]] stores #T(fr_world, fr_voxel).
        - Extent: #code-inline[voxel_extent: Tensor[#(symb.shape.B), 6]] in voxel frame (meters).
        - Voxel centers (required in v3): #code-inline[pts_world: Tensor[#(symb.shape.B), #(symb.shape.Vvox), 3]] (world frame).
        - Grid resolution in our runs: #symb.shape.D x #symb.shape.H x #symb.shape.Wdim (typically 48^3).
      ]
    ],
    [
      #figure(
        evl-backbone-tree-grid(node-width: 15em, text-size: 8.5pt, spacing: (8pt, 12pt)),
        caption: [EVL grid contract outputs (schematic).],
      )
    ],
  )
]

#slide(title: [EVL backbone: evidence tensors])[
  #grid(
    [
      #color-block(title: [Observation evidence (aligned with NBV intuition)])[
        - Occupied evidence: #symb.vin.occ_in shape #code-inline[Tensor[#(symb.shape.B), 1, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]].
        - Coverage proxy: #symb.vin.counts shape #code-inline[Tensor[#(symb.shape.B), #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]].
        - Optional free-space evidence: #code-inline[free_input] shape #code-inline[Tensor[#(symb.shape.B), 1, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]].
        - VIN normalizes #code-inline[counts] #sym.arrow.r #code-inline[counts_norm in [0,1]] to derive observed/unknown priors.
        - If #code-inline[free_input] is missing, VIN v3 derives it as #code-inline[observed * (1 - occ_input)].
      ]
    ],
    [
      #figure(
        evl-backbone-tree-evidence(node-width: 15em, text-size: 8.5pt, spacing: (8pt, 12pt)),
        caption: [EVL evidence tensors (schematic).],
      )
    ],
  )
]

#slide(title: [EVL backbone: head outputs (available vs used)])[
  #grid(
    [
      #color-block(title: [Head outputs (dense voxel heads)])[
        - Occupancy probability: #symb.vin.occ_pr shape #code-inline[Tensor[#(symb.shape.B), 1, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]].
        - Centerness: #symb.vin.cent_pr shape #code-inline[Tensor[#(symb.shape.B), 1, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] (VIN v3 uses #symb.vin.cent_pr_nms).
          - VIN v3 uses the NMS variant: #symb.vin.cent_pr_nms.
        - Optional 3D detection heads:
          + #code-inline[bbox_pr: Tensor[B, 7, D, H, W]] (OBB regression channels).
          + #code-inline[clas_pr: Tensor[B, K, D, H, W]] (class logits/probabilities).
          + #code-inline[obbs_pr_nms: ObbTW[B, M, 34]] (post-NMS boxes).
      ]

      #color-block(title: [What VIN v3 caches/uses])[
        - Used to build #symb.vin.field_v: {#symb.vin.occ_pr, #symb.vin.occ_in, #symb.vin.counts_norm, #symb.vin.cent_pr_nms}
          (+ optional #symb.vin.free_in, #symb.vin.unknown, #symb.vin.new_surface_prior via #code-inline[scene_field_channels]).
        - Not used in the baseline scorer (for now): dense box/class heads and post-processed OBBs.
        - Optional richer features (available in EVL but typically not cached): #code-inline[voxel_feat], #code-inline[occ_feat], #code-inline[obb_feat].
      ]
    ],
    [
      #figure(
        evl-backbone-tree-heads(node-width: 15em, text-size: 8.5pt, spacing: (8pt, 12pt)),
        caption: [EVL head voxel fields (occ vs obb).],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// VIN v3 forward: feature branches (tensors + frames + shapes)
// ---------------------------------------------------------------------------

#slide(title: [VIN v3 forward pass: frames + shape legend])[
  #grid(
    [
      #color-block(title: [Frames used in v3])[
        - #code-inline[w]: world (ASE global, meters).
        - #code-inline[r]: reference rig frame at the snippet reference time (rig_ref).
        - #code-inline[q]: candidate camera frame (one per candidate).
        - #code-inline[v]: EVL voxel grid frame (axis-aligned metric grid).
        - #code-inline[s]: screen/pixel space (PyTorch3D projection output).
      ]
      #color-block(title: [PoseTW convention])[
        - Poses are stored as world #sym.arrow.l frame transforms, e.g.
          #T(fr_world, fr_cam), #T(fr_world, fr_rig_ref), #T(fr_world, fr_voxel).
        - Relative candidate pose:
          $#T(fr_rig_ref, fr_cam) = #T(fr_world, fr_rig_ref)^(-1) dot #T(fr_world, fr_cam).$
        - Points are always expressed explicitly as #code-inline[x_w], #code-inline[x_r], #code-inline[x_s].
      ]
    ],
    [
      #color-block(title: [Shape symbols (used below)])[
        - Batch size: #symb.shape.B. Candidates: #symb.shape.Nq. Trajectory length: #symb.shape.Tlen.
        - Voxel grid: #symb.shape.D x #symb.shape.H x #symb.shape.Wdim. Voxel centers: #symb.shape.Vvox = #symb.shape.D x #symb.shape.H x #symb.shape.Wdim.
        - Pooled voxel points: #symb.shape.Pproj = #symb.shape.Gpool^3.
        - Semidense points: #symb.shape.P (padded) and #symb.shape.Pfr (#sym.arrow.r subsampled for projection).
      ]
      #good-note(width: 100%)[
        In code: see `oracle_rri/oracle_rri/vin/model_v3.py::_forward_impl` for the exact tensor flow.
      ]
    ],
  )
]

#slide(title: [Branch 0: inputs #sym.arrow.r PreparedInputs])[
  #grid(
    [
      #color-block(title: [Inputs (from VinOracleBatch)])[
        - Candidate poses: #code-inline[PoseTW[#(symb.shape.B), #(symb.shape.Nq), 12]] (world #sym.arrow.l cam_q).
        - Reference pose: #code-inline[PoseTW[#(symb.shape.B), 12]] (world #sym.arrow.l rig_ref).
        - Cameras: #code-inline[PerspectiveCameras] with flat batch size #code-inline[B x N_q].
          - Ex: #code-inline[R[B x N_q, 3, 3]], #code-inline[T[B x N_q, 3]], intrinsics, #code-inline[image_size[B x N_q, 2]].
        - EVL backbone output (cached): #code-inline[t_world_voxel: PoseTW[#(symb.shape.B), 12]], #code-inline[voxel_extent[#(symb.shape.B), 6]], #code-inline[pts_world[#(symb.shape.B), #(symb.shape.Vvox), 3]].
      ]
    ],
    [
      #color-block(title: [PreparedInputs (after normalization)])[
        - #code-inline[pose_world_cam: PoseTW[#(symb.shape.B), #(symb.shape.Nq), 12]] stores #T(fr_world, fr_cam).
        - #code-inline[pose_world_rig_ref: PoseTW[#(symb.shape.B), 12]] stores #T(fr_world, fr_rig_ref).
        - #code-inline[t_world_voxel: PoseTW[#(symb.shape.B), 12]] stores #T(fr_world, fr_voxel).
        - Transforms/modules:
          #code-inline[ensure_candidate_batch] + #code-inline[ensure_pose_batch] (broadcast to B),
          optional #code-inline[rotate_yaw_cw90(undo=True)] for pose convention alignment.
        - Optional CW90 undo (poses) must be consistent with #code-inline[p3d_cameras] correction.
      ]
    ],
  )
]

#slide(title: [Branch 1: PoseFeatures (rig-relative pose encoding)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - #code-inline[pose_world_cam] stores #T(fr_world, fr_cam) and #code-inline[pose_world_rig_ref] stores #T(fr_world, fr_rig_ref).
        - Frame transform: $#T(fr_rig_ref, fr_cam) = #T(fr_world, fr_rig_ref)^(-1) dot #T(fr_world, fr_cam)$.
        - Transforms/modules: R6D + LFF + MLP, i.e. #code-inline[pose_vec] #sym.arrow.r #code-inline[LFF] #sym.arrow.r #code-inline[MLP].
      ]
      #color-block(title: [Outputs])[
        - #code-inline[pose_vec: Tensor[#(symb.shape.B), #(symb.shape.Nq), 9]] (t_r_cq + R6D(R_r_cq)).
        - #code-inline[pose_enc: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fpose)]] (LFF + MLP).
        - #code-inline[candidate_center_rig_m: Tensor[#(symb.shape.B), #(symb.shape.Nq), 3]] (camera center in rig_ref, meters).
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - Pose features are expressed in #code-inline[r] (reference rig frame).
        - This removes global-frame drift and makes candidate comparisons stable.
      ]
    ],
  )
]

#slide(title: [Branch 2: FieldBundle (EVL voxel scene field)])[
  #grid(
    [
      #color-block(title: [Inputs (EVL tensors)])[
        - #code-inline[occ_pr, occ_input, cent_pr_nms: Tensor[#(symb.shape.B), 1, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]].
        - #code-inline[counts: Tensor[#(symb.shape.B), #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] (observation counts).
        - #code-inline[free_input] if available else derived from observed/free evidence.
        - Derived channels (scalars on voxel grid):
          #code-inline[counts_norm = log1p(counts) / log1p(max(counts))],
          #code-inline[unknown = 1 - counts_norm],
          #code-inline[new_surface_prior = unknown dot.o occ_pr].
        - Channel selection is configured via #code-inline[scene_field_channels].
      ]
      #color-block(title: [Outputs])[
        - #code-inline[field_in: Tensor[#(symb.shape.B), #(symb.shape.Fin), #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] (selected channels).
        - #code-inline[field: Tensor[#(symb.shape.B), #(symb.shape.Fg), #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] (Conv3d #sym.arrow.r GN #sym.arrow.r GELU).
        - #code-inline[aux] dict includes #code-inline[counts_norm, observed, unknown, new_surface_prior].
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - The scene field lives on the EVL voxel grid (#code-inline[v]) and is indexed by voxel cells.
        - World-space voxel centers #code-inline[pts_world] provide geometry for positional keys and projections.
      ]
    ],
  )
]

#slide(title: [Branch 3: voxel_valid_frac (candidate center coverage proxy)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - Candidate camera centers: #code-inline[x_w_cq = pose_world_cam.t] as #code-inline[Tensor[#(symb.shape.B), #(symb.shape.Nq), 3]] in world frame.
        - #code-inline[counts_norm] from FieldBundle (normalized observation count per voxel).
        - #code-inline[t_world_voxel] stores #T(fr_world, fr_voxel) and #code-inline[voxel_extent] maps voxel coords (metres) to grid indices.
        - Transform: sample #code-inline[counts_norm] at x_w_cq via #code-inline[sample_voxel_field] (pc_to_vox + sample_voxels).
      ]
      #color-block(title: [Outputs])[
        - #code-inline[voxel_valid_frac: Tensor[#(symb.shape.B), #(symb.shape.Nq)]] in [0, 1].
        - Intuition: how much voxel evidence exists at the candidate camera center.
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - Input centers are in #code-inline[w] and are sampled in the voxel grid (#code-inline[v]) via #T(fr_world, fr_voxel) and voxel_extent.
        - Output is a scalar coverage proxy (no frame).
      ]
    ],
  )
]

#slide(title: [Branch 4: GlobalContext (pos_grid + global_feat)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - #code-inline[field: Tensor[#(symb.shape.B), #(symb.shape.Fg), #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] (#code-inline[v] grid).
        - #code-inline[pose_enc: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fpose)]] (#code-inline[r] features).
        - #code-inline[pts_world: Tensor[#(symb.shape.B), #(symb.shape.Vvox), 3]] voxel centers in world frame.
        - Transform chain: voxel centers in #code-inline[w] are expressed in rig_ref via $#T(fr_world, fr_rig_ref)^(-1)$.
        - Modules: #code-inline[pos_grid_from_pts_world] + #code-inline[PoseConditionedGlobalPool] (pool3d + MHA + MLP).
      ]
      #color-block(title: [Outputs])[
        - #code-inline[pos_grid: Tensor[#(symb.shape.B), 3, #(symb.shape.D), #(symb.shape.H), #(symb.shape.Wdim)]] normalized voxel centers in rig_ref.
        - #code-inline[global_feat: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fg)]] pose-conditioned global vector per candidate.
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - #code-inline[pos_grid] is in #code-inline[r] (rig_ref) and normalized by voxel extent.
        - #code-inline[global_feat] is a per-candidate feature (no spatial frame) but conditioned on #code-inline[r] pose and #code-inline[r] positional keys.
      ]
    ],
  )
]

#slide(title: [Branch 5: voxel_proj + FiLM (projection of voxel centers)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - Pooled voxel centers: #code-inline[voxel_points: Tensor[#(symb.shape.B), #(symb.shape.Pproj), 3]] in world frame.
        - Cameras: #code-inline[p3d_cameras] aligned with candidates (flat #code-inline[B x N_q]).
        - Modules: #code-inline[transform_points_screen] (projection) + #code-inline[\_encode_semidense_projection_features] (stats; uniform weights) + FiLM.
      ]
      #color-block(title: [Outputs])[
        - Projection data: #code-inline[x_s, y_s, z_s, valid: Tensor[B x N_q, #(symb.shape.Pproj)]] in screen space.
        - Scalar projection stats: #code-inline[voxel_proj: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fproj)]] with
          (#code-inline[coverage], #code-inline[empty_frac], #code-inline[valid_frac], #code-inline[depth_mean], #code-inline[depth_std]).
        - FiLM modulation: #code-inline[(gamma, beta) = Linear(voxel_proj)] then #code-inline[global_feat] #sym.arrow.r #code-inline[global_feat_film]:
          #block[#align(center)[#eqs.features.film]]
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - #code-inline[voxel_points] are in #code-inline[w]; #code-inline[x_s, y_s, z_s] are in #code-inline[s] (pixels + depth).
        - #code-inline[voxel_proj] is a per-candidate scalar summary (no frame).
      ]
    ],
  )
]

#slide(title: [Branch 6: semidense_proj (scalar projection statistics)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - Semidense points: #code-inline[points_world: Tensor[#(symb.shape.B), #(symb.shape.Pfr), 3 + 2]] in world frame:
          (#code-inline[x_w, y_w, z_w, 1/sigma_d, n_obs]).
          - (code aliases: #code-inline[inv_dist_std], #code-inline[obs_count])
        - Cameras: #code-inline[p3d_cameras] aligned with candidates.
        - Modules:
          + random subsampling with #code-inline[lengths] mask (#code-inline[semidense_proj_max_points] budget),
          + projection (#code-inline[transform_points_screen]),
          + reliability-weighted aggregation (uses #code-inline[n_obs] and #code-inline[1/sigma_d]).
      ]
      #color-block(title: [Outputs])[
        - Projection data: #code-inline[x_s, y_s, z_s, valid: Tensor[B x N_q, #(symb.shape.Pfr)]].
        - Scalar features: #code-inline[semidense_proj: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fproj)]]:
          (#code-inline[coverage], #code-inline[empty_frac], #code-inline[semidense_candidate_vis_frac], #code-inline[depth_mean], #code-inline[depth_std]).
        - #code-inline[semidense_candidate_vis_frac] is also exported as a prediction field for diagnostics/weighting.
        - Feature definitions:
          + #code-inline[coverage] / #code-inline[empty_frac]: occupancy of a $G times G$ screen grid (G=#wb.vin_effective.semidense_proj_grid_size),
          + #code-inline[semidense_candidate_vis_frac]: reliability-weighted valid/finite ratio (see #eqs.features.semidense_visibility),
          + #code-inline[depth_mean], #code-inline[depth_std]: reliability-weighted moments of camera-axis depth $z$ over valid projections.
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - Input points are in #code-inline[w]; intermediate projections are in #code-inline[s]; output is per-candidate scalar.
        - Reliability weighting uses #code-inline[inv_dist_std] (1/sigma_d) and #code-inline[obs_count] (track length), normalized by
          #code-inline[semidense_inv_dist_std_min/p95] and #code-inline[semidense_obs_count_max] (clamped to [0,1]).
      ]
    ],
  )
]

#slide(title: [Semidense projection: transforms + grid maps])[
  #grid(
    columns: (1.25fr, 1fr),
    gutter: 0.4cm,
    [
      #grid(
        rows: (1fr, auto),
        gutter: 0.25cm,
        [
          #color-block(title: [Transforms + aggregation], spacing: 0.45em)[
            #set text(size: 15pt)
            - Inputs:
              + #code-inline[points_world: Tensor[#(symb.shape.B), #(symb.shape.Pfr), 5]].
              + #code-inline[p3d_cameras: PerspectiveCameras] with #code-inline[image_size: Tensor[B x N_q, 2]] = (#(symb.shape.H), #(symb.shape.Wdim)).
            - Projection (Pytorch3D #code-inline[transform_points_screen]):
            #block[#align(center)[$
              X_c = R X_w + T,
              quad
              u = f_x X_(c,x) / X_(c,z) + c_x,
              quad
              v = f_y X_(c,y) / X_(c,z) + c_y
            $]]
            - Validity mask:
            #block[#align(center)[#eqs.features.semidense_validity]]
            - Grid binning (G = #code-inline[semidense_proj_grid_size]):
            #block[#align(center)[$
              u_"bin" = "clamp"(u / W dot G, 0, G-1),
              quad
              v_"bin" = "clamp"(v / H dot G, 0, G-1)
            $]]
            - Per-cell maps (shape #code-inline[Tensor[B x N_q, G, G]]):
            #block[#align(center)[$
              "counts"_g = sum_j bb(1)["m"_(i,j)] dot bb(1)["bin"_(i,j) = g]
            $]]
            #block[#align(center)[$
              "weights"_g = sum_j w_(i,j) dot bb(1)["m"_(i,j)] dot bb(1)["bin"_(i,j) = g]
            $]]
            #block[#align(center)[$
              "depth_mean"_g =
              (sum_j w_(i,j) z_(i,j) bb(1)["m"_(i,j)]) /
              (sum_j w_(i,j) bb(1)["m"_(i,j)])
            $]]
            - Aggregation intuition: if H=W=120 and G=12 (offline_only), each cell covers ~10x10 pixels.
          ]
        ],
        [
          #color-block(title: [Reliability weights (per point)], spacing: 0.45em)[
            #set text(size: 15pt)
            #block[#align(center)[$
              w_(i,j) = a_(i,j) dot b_(i,j)
            $]]
            #block[#align(center)[$
              a_(i,j) =
              "clamp"(("log"(1 + n_"obs")) / ("log"(1 + n_"obs,max")), 0, 1)
            $]]
            #block[#align(center)[$
              b_(i,j) =
              "clamp"(("inv"_"dist,std" - "inv"_"min") / ("inv"_"p95" - "inv"_"min"), 0, 1)
            $]]
            - #code-inline[n_obs,max] = #code-inline[semidense_obs_count_max],
              #code-inline[inv_min] = #code-inline[semidense_inv_dist_std_min],
              #code-inline[inv_p95] = #code-inline[semidense_inv_dist_std_p95].
            - Coverage stats:
            #block[#align(center)[$
              "coverage" = "mean"("counts" > 0),
              quad
              "empty_frac" = 1 - "coverage"
            $]]
          ]
        ],
      )
    ],
    [
      #figure(
        caption: [Semidense projection maps (counts / weights / depth std).],
      )[
        #grid(
          columns: (1fr, 1fr, 1fr),
          gutter: 0.2cm,
          [
            #image(fig_path + "app-paper/semi-dense-counts-proj.png", width: 100%)
          ],
          [
            #image(fig_path + "app-paper/semi-dense-weight-proj.png", width: 100%)
          ],
          [
            #image(fig_path + "app-paper/semi-dense-std-proj.png", width: 100%)
          ],
        )
      ]
    ],
  )
]

#slide(title: [Branch 7: semidense_grid_feat (tiny CNN on a projection grid)])[
  #grid(
    [
      #color-block(title: [How it differs from semidense_proj])[
        - #code-inline[\_encode_semidense_projection_features]: reduces all points to 5 scalars per candidate (no spatial layout).
        - #code-inline[\_encode_semidense_grid_features]: builds an explicit G x G grid in screen space and runs a tiny CNN.
        - Motivation: capture *where* points land in the image plane (coarse structure), not only how many.
      ]
      #color-block(title: [Inputs / outputs])[
        - Inputs: same #code-inline[proj_data] as above (#code-inline[x_s, y_s, z_s, valid]).
        - Build grid channels: occupancy, depth_mean, depth_std:
          #code-inline[grid: Tensor[B x N_q, 3, #(symb.shape.Gproj), #(symb.shape.Gproj)]] (screen-space bins).
        - Modules: #code-inline[scatter_add] binning #sym.arrow.r Conv2d #sym.arrow.r GELU #sym.arrow.r Conv2d #sym.arrow.r GELU #sym.arrow.r GAP #sym.arrow.r Linear.
        - CNN output: #code-inline[semidense_grid_feat: Tensor[#(symb.shape.B), #(symb.shape.Nq), F_cnn]] appended to the head.
        - Config: enabled by #code-inline[semidense_cnn_enabled] with #code-inline[semidense_cnn_channels] and #code-inline[semidense_cnn_out_dim].
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - The grid lives in #code-inline[s] (screen/pixels binned to a fixed grid).
        - Output is a per-candidate vector (no frame).
        - Current implementation uses #code-inline[valid] masking (no #code-inline[n_obs] / #code-inline[1/sigma_d] weighting inside the grid).
        - The diagnostics UI renders the exact grid maps fed to the CNN (occupancy/mean/std).
      ]
    ],
  )
]

#slide(title: [Branch 8: trajectory context (traj_feat + traj_ctx)])[
  #grid(
    [
      #color-block(title: [Inputs])[
        - Trajectory: #code-inline[t_world_rig: PoseTW[#(symb.shape.B), #(symb.shape.Tlen), 12]] (world #sym.arrow.l rig_t).
        - Reference rig pose stores #T(fr_world, fr_rig_ref).
        - Convert per frame: $#T(fr_rig_ref, fr_rig) = #T(fr_world, fr_rig_ref)^(-1) dot #T(fr_world, fr_rig)$.
        - Modules: #code-inline[TrajectoryEncoder] (R6D + LFF + MLP) and optional #code-inline[traj_attn] (MHA).
      ]
      #color-block(title: [Outputs])[
        - Per-frame encodings: #code-inline[traj_pose_vec: Tensor[#(symb.shape.B), #(symb.shape.Tlen), D_v]] and #code-inline[traj_pose_enc: Tensor[#(symb.shape.B), #(symb.shape.Tlen), #(symb.shape.Ftau)]].
        - Pooled embedding: #code-inline[traj_feat: Tensor[#(symb.shape.B), #(symb.shape.Ftau)]].
        - Optional attention: #code-inline[traj_ctx: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fpose)]] (pose queries attend to trajectory keys/values).
      ]
    ],
    [
      #color-block(title: [Frame summary])[
        - All trajectory encodings are expressed relative to #code-inline[r] (rig_ref).
        - This provides a "stage" signal: what has already been observed along the path.
      ]
    ],
  )
]

#slide(title: [Branch 9: head input concat + VinPrediction outputs])[
  #grid(
    [
      #color-block(title: [Final scorer input (per candidate)])[
        - Concatenate:
          #code-inline[pose_enc] (#code-inline[r]-relative),
          #code-inline[global_feat] (scene context),
          #code-inline[semidense_proj] (scalar view evidence),
          optionally #code-inline[semidense_grid_feat] and #code-inline[traj_ctx].
        - Result: #code-inline[feats: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.Fhead)]].
        - Modules: #code-inline[head_mlp] (Linear #sym.arrow.r GELU #sym.arrow.r Dropout) + #code-inline[CoralLayer].
      ]
      #color-block(title: [Outputs (VinPrediction)])[
        - #code-inline[logits: Tensor[#(symb.shape.B), #(symb.shape.Nq), K-1]] (CORAL thresholds).
        - #code-inline[prob: Tensor[#(symb.shape.B), #(symb.shape.Nq), #(symb.shape.K)]] (class marginals).
        - #code-inline[expected, expected_normalized: Tensor[#(symb.shape.B), #(symb.shape.Nq)]] (ranking proxy).
        - #code-inline[candidate_valid: Tensor[#(symb.shape.B), #(symb.shape.Nq)]] diagnostic mask (finite pose + nonzero coverage proxies).
        - #code-inline[voxel_valid_frac] and #code-inline[semidense_candidate_vis_frac] are logged and can reweight loss (coverage schedule).
      ]
    ],
    [
      #good-note(width: 100%)[
        Training uses CORAL loss on #code-inline[logits] and logs ranking metrics on #code-inline[expected_normalized].
        Coverage-weight schedules (train only) can reweight per-candidate losses using #code-inline[voxel_valid_frac] or #code-inline[semidense_candidate_vis_frac].
      ]
    ],
  )
]

#slide(title: [Pose encoding and rig-relative conditioning])[
  #grid(
    [
      #color-block(title: [Rig-relative pose encoding])[
        - Candidate pose in reference rig frame:
          $#T(fr_rig_ref, fr_cam) = #T(fr_world, fr_rig_ref)^(-1) dot #T(fr_world, fr_cam).$
        - Encode translation + rotation with 6D representation + Learnable Fourier Features.
        - Motivation: continuity + stable gradients for view-conditioned ranking.
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_pose_descriptor.png", width: 100%),
        caption: [Pose descriptor diagnostics (implementation).],
      )
    ],
  )
]

#slide(title: [Global context + FiLM modulation])[
  #grid(
    [
      #color-block(title: [Pose-conditioned global context])[
        - Pool the EVL scene field to $G_"pool"^3$ tokens (G_pool=#wb.vin_effective.global_pool_grid_size).
        - Add positional encoding of voxel centers and cross-attend with pose embeddings as queries.
        - Output a per-candidate global vector #symb.vin.global (candidate-conditioned scene evidence).
      ]
      #color-block(title: [FiLM from projection stats])[
        - Project pooled voxel centers into each candidate view and summarize screen-space coverage + depth stats.
        - Predict per-channel $(#(symb.vin.gamma), #(symb.vin.beta))$ and modulate:
        #block[#align(center)[#eqs.features.film]]
      ]
    ],
    [
      #figure(
        image(fig_path + "app/scene_field_occ_pr.png", width: 100%),
        caption: [Scene-field diagnostics (EVL occupancy prior slice).],
      )
    ],
  )
]

#slide(title: [Semidense projections: visibility + grid CNN])[
  #grid(
    [
      #color-block(title: [Screen-space validity + visibility])[
        - Project semidense points into each candidate camera.
        - Valid if finite, $z>0$, and inside image bounds:
        #block[#align(center)[#eqs.features.semidense_validity]]
        - Visibility fraction is a reliability-weighted valid/finite ratio:
        #block[#align(center)[#eqs.features.semidense_visibility]]
        - Reliability weights $w_(i,j)$ combine per-point track length and depth uncertainty:
        #block[#align(center)[$
          w_(i,j) =
          "clamp"(("log"(1 + n_"obs")) / ("log"(1 + n_"obs,max")), 0, 1)
          dot
          "clamp"(("inv"_"dist,std" - "inv"_"min") / ("inv"_"p95" - "inv"_"min"), 0, 1)
        $]]
        - In code: $n_"obs,max"$=#code-inline[semidense_obs_count_max], $"inv"_"min"$=#code-inline[semidense_inv_dist_std_min],
          $"inv"_"p95"$=#code-inline[semidense_inv_dist_std_p95] (global cache summaries).
        - The scalar semidense branch exports:
          #code-inline[semidense_proj = (coverage, empty_frac, semidense_candidate_vis_frac, depth_mean, depth_std)].
      ]
      #color-block(title: [Grid features (local view-plane structure)])[
        - Bin valid projections into a $G_"sem" times G_"sem"$ grid (G_sem=#wb.vin_effective.semidense_proj_grid_size).
        - Per-bin maps (screen-space): occupancy, mean depth, depth std (unweighted; valid-only).
        - Tiny CNN encodes these 3 maps into a compact per-candidate feature appended to the head.
        - Diagnostics UI shows these same grids (“Semidense CNN grid inputs”).
        - Diagnostics also renders per-cell maps with reliability weighting (counts/weights/depth_mean/depth_std) to interpret the scalar stats.
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_shell_descriptor_concept.png", width: 100%),
        caption: [Concept: per-candidate semidense projection descriptor.],
      )
    ],
  )
]

#slide(title: [Trajectory encoder (optional, but historically strong)])[
  #let m = wb.metrics
  #grid(
    [
      #color-block(title: [What it encodes])[
        - Input: rig pose history #code-inline[PoseTW[#(symb.shape.B), #(symb.shape.Tlen), 12]] from `VinSnippetView.t_world_rig`.
        - Output: a compact trajectory feature (#(symb.shape.Ftau)) appended to the scorer input.
        - Motivation: RRI depends on "what has already been seen" (stage), not only the current scene field.
      ]
      #color-block(title: [Evidence from our runs])[
        - Optuna top-trial pattern: trajectory encoder is enabled in many high-performing trials.
        - In the best v3 run, the trajectory path carries non-trivial gradient energy:
          #code-inline[grad_norm_traj_encoder]=#round(m.at("train-gradnorms/grad_norm_traj_encoder"), digits: 2),
          #code-inline[grad_norm_traj_attn]=#round(m.at("train-gradnorms/grad_norm_traj_attn"), digits: 2).
      ]
    ],
    [
      #figure(
        image(fig_path + "app/traj.png", width: 100%),
        caption: [Trajectory diagnostics (Streamlit): historical rig poses and candidate views.],
      )
    ],
  )
]

#slide(title: [Scoring head: feature concat + CORAL bins])[
  #let v = wb.vin_effective
  #grid(
    [
      #color-block(title: [Scorer input (per candidate)])[
        - Concatenate candidate-specific and scene context features:
          + pose encoding,
          + global voxel context #symb.vin.global,
          + semidense projection stats and grid CNN embedding,
          + optional trajectory embedding (if enabled).
        - Head MLP hyperparameters (effective config):
          #code-inline[hidden_dim]=#v.at("head_hidden_dim"),
          #code-inline[num_layers]=#v.at("head_num_layers"),
          #code-inline[dropout]=#round(v.at("head_dropout"), digits: 3).
      ]
      #color-block(title: [Ordinal output])[
        - Predict cumulative probabilities with CORAL and convert to class marginals / expectation:
        #block[#align(center)[#eqs.coral.marginals]]
        #block[#align(center)[#eqs.coral.expected]]
      ]
    ],
    [
      #figure(
        image(fig_path + "impl/vin/vin_rri_binning.png", width: 100%),
        caption: [RRI binning / thresholds used for ordinal supervision.],
      )
    ],
  )
]

// ---------------------------------------------------------------------------
// Objective + metrics
// ---------------------------------------------------------------------------

#section-slide(
  title: [Objective and Metrics],
  subtitle: [Ordinal supervision + diagnostics-first monitoring],
)

#slide(title: [Oracle target + CORAL objective])[
  #grid(
    [
      #color-block(title: [Oracle RRI definition])[
        #block[#align(center)[#eqs.rri.rri]]
        - RRI is the relative reduction in point-to-mesh error after adding a candidate view.
        - We log accuracy (#symb.oracle.points #sym.arrow.r #symb.ase.mesh) and completeness (#symb.ase.mesh #sym.arrow.r #symb.oracle.points) components.
      ]
      #color-block(title: [CORAL ordinal loss])[
        #block[#align(center)[#eqs.coral.loss]]
        #block[#align(center)[#eqs.coral.marginals]]
        #block[#align(center)[#eqs.coral.expected]]
      ]
    ],
    [
      #color-block(title: [Scheduled coverage reweighting (train only)])[
        - Coverage proxy $c_i in [0, 1]$ from voxel validity or semidense visibility.
        - Curriculum-style anneal of coverage strength to reduce early gradient variance.
        #block[#align(center)[#eqs.coverage.weight]]
        #block[#align(center)[#eqs.coverage.weighted_loss]]
      ]
    ],
  )
]

#slide(title: [Logged diagnostics (selected)])[
  #grid(
    [
      #color-block(title: [Ranking quality])[
        - Spearman correlation (pred expected RRI vs oracle RRI).
        - Top-3 accuracy (does true label fall into top-3 predicted bins?).
        - Confusion matrices and label histograms.
      ]
      #color-block(title: [Evidence + reliability])[
        - Voxel validity fraction (coverage proxy).
        - Semidense candidate visibility fraction (per-candidate projection validity).
        - Candidate_valid fraction (finite pose and evidence present).
      ]
    ],
    [
      #color-block(title: [Optimization diagnostics])[
        - Learning rate (#code-inline[lr-AdamW]).
        - Per-module grad norms (#code-inline[train-gradnorms/...] keys).
        - CORAL monotonicity violation rate (ordinal sanity check).
      ]
    ],
  )
]

#slide(title: [Metric definitions (logged keys)])[
  #grid(
    [
      #color-block(title: [Ranking metrics])[
        - Spearman:
        #block[#align(center)[#eqs.metrics.spearman]]
        - Top-k accuracy:
        #block[#align(center)[#eqs.metrics.topk_acc]]
        - Confusion matrix:
        #block[#align(center)[#eqs.metrics.confusion]]
      ]
      #color-block(title: [Evidence / validity])[
        - Candidate validity (example predicate):
        #block[#align(center)[#eqs.metrics.candidate_validity]]
        - Voxel validity: coverage proxy derived from EVL observation counts.
        - Semidense visibility: $v_i^("sem")$ (see previous projection slide).
      ]
    ],
    [
      #color-block(title: [Loss + optimization])[
        - CORAL relative-to-random baseline:
        #block[#align(center)[#eqs.coral.rel_random]]
        - Module grad norms:
        #block[#align(center)[#eqs.metrics.grad_norm]]
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Streamlit diagnostics (geometry-first)
// ---------------------------------------------------------------------------

#section-slide(
  title: [Streamlit Diagnostics],
  subtitle: [Geometry checks and failure-mode discovery],
)

#slide(title: [Diagnostics gallery])[
  #figure(
    grid(
      rows: (1fr, 1fr),
      image(fig_path + "app/render_frusta.png", width: 100%),
      image(fig_path + "app/depth_hist.png", width: 100%),
      image(fig_path + "app/semidense.png", width: 100%),
      image(fig_path + "app/depth_render.png", width: 100%),
    ),
    caption: [Streamlit diagnostics: frusta, depth histograms, semi-dense overlays, and depth renders.],
  )

  #color-block(title: [Why diagnostics matter])[
    - Validate coordinate conventions and candidate visibility.
    - Catch render failures (empty z-buffers, wall look-through, degenerate poses).
    - Inspect candidate distributions before training a scorer.
  ]
]

#slide(title: [Common failure modes (and the checks we use)])[
  #grid(
    [
      #color-block(title: [Geometry / frame consistency])[
        - CW90 correction mismatch between poses and PyTorch3D cameras can destroy projections.
          - For cached data: keep #code-inline[apply_cw90_correction]=false unless corrected in lockstep.
        - Screen projection conventions (NDC vs pixel space, image_size ordering) can make almost all points invalid.
        - Sanity checks: per-candidate valid ratios, visibility fractions, and frustum overlays.
      ]
      #color-block(title: [Data / padding pitfalls])[
        - Padded semidense points must be masked using #code-inline[lengths] (avoid non-finite XYZ from padding).
        - Label binning depends on candidate sampling; changing the distribution requires re-binning or re-training.
      ]
    ],
    [
      #color-block(title: [Training stability signals])[
        - If candidate-specific features do not reach the head, the model collapses to a near-constant predictor.
        - Overly strict masks (candidate_valid) reduce gradient signal; track #code-inline[candidate_valid_frac].
        - Use confusion matrices, label histograms, and grad norms to detect collapse early.
      ]
      #color-block(title: [Why this matters])[
        - Most "bad training runs" can be traced back to invalid geometry evidence rather than optimizer bugs.
        - The Streamlit panel is the fastest way to localize these issues.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Empirical evidence: Optuna + W&B
// ---------------------------------------------------------------------------

#section-slide(
  title: [Empirical Evidence],
  subtitle: [Optuna sweep patterns + best W&B run diagnostics],
)

#slide(title: [Optuna sweep: what correlates with good trials?])[
  #let n_trials = top_trials.len()
  #let n_traj = top_trials.filter(r => r.traj == "T").len()
  #let n_frustum = top_trials.filter(r => r.frustum == "T").len()
  #let n_vfeat = top_trials.filter(r => r.vfeat == "T").len()
  #let n_vgate_off = top_trials.filter(r => r.vgate == "F").len()

  #grid(
    [
      #color-block(title: [Top trials summary])[
        Among the top #n_trials trials in #code-inline[vin-v2-sweep]:
        - trajectory encoder enabled: #n_traj / #n_trials
        - semidense frustum attention enabled: #n_frustum / #n_trials
        - voxel-validity features included: #n_vfeat / #n_trials
        - voxel gate disabled: #n_vgate_off / #n_trials
      ]
      #color-block(title: [Interpretation (caveat)])[
        - The study is partly non-stationary (config corrections in early trials).
        - Still, candidate-specific modules (traj/frustum) appear consistently helpful.
      ]
    ],
    [
      #figure(
        image(fig_path + "vin_v2/optuna_objective_vs_trial.png", width: 100%),
        caption: [Optuna objective vs trial index (vin-v2-sweep).],
      )
    ],
  )
]

#slide(title: [Optuna toggle plots (phase-mixed; interpret cautiously)])[
  #grid(
    [
      #figure(
        image(fig_path + "vin_v2/optuna_objective_vs_use_point_encoder.png", width: 100%),
        caption: [Objective vs `use_point_encoder` (phase-mixed).],
      )
    ],
    [
      #figure(
        image(fig_path + "vin_v2/optuna_objective_vs_enable_semidense_frustum.png", width: 100%),
        caption: [Objective vs `enable_semidense_frustum` (weak positive trend).],
      )
    ],
  )

  #color-block(title: [Interpretation])[
    - Toggle evidence is confounded by phase mixing.
    - Next sweep should be stationary: architectural toggles only (fixed training regime).
  ]
]

#slide(title: [Mode collapse case study: vin-v3-01 vs T41])[
  #let a = v3_vs_t41.vin_v3_01
  #let b = v3_vs_t41.t41

  #grid(
    [
      #color-block(title: [Symptom summary (imported)])[
        #figure(
          kind: "table",
          supplement: [Table],
          caption: [Selected metrics highlighting the collapse in #code-inline[vin-v3-01] vs the best optuna run #code-inline[T41].],
          text(size: 9pt)[
            #table(
              columns: (15em, 1fr, 1fr),
              align: (left, right, right),
              toprule(),
              table.header([Metric], [vin-v3-01], [T41]),
              midrule(), [train spearman], [#round(a.train_spearman, digits: 3)],
              [#round(b.train_spearman, digits: 3)], [train top-3 acc], [#round(a.train_top3_accuracy, digits: 3)],
              [#round(b.train_top3_accuracy, digits: 3)],
              [candidate_valid_frac],
              [#round(a.candidate_valid_frac, digits: 3)],

              [#round(b.candidate_valid_frac, digits: 3)],
              [semidense vis (train)],
              [#round(a.semidense_candidate_vis_frac_mean_train, digits: 3)],

              [#round(b.semidense_candidate_vis_frac_mean_train, digits: 3)],
              [coverage_weight_strength],
              [#round(a.coverage_weight_strength, digits: 3)],

              [#round(b.coverage_weight_strength, digits: 3)], bottomrule(),
            )
          ],
        )
      ]
    ],
    [
      #color-block(title: [Likely causes (from our analysis)])[
        - Candidate-specific evidence did not reach the scorer in early v3 versions (semidense stats were computed but only used for masking).
        - Overly strict validity masks and voxel gating reduce effective gradients and amplify collapse.
        - Training regime differences (LR schedule and gradient energy) change how easily the model escapes a near-constant predictor.
      ]
      #color-block(title: [What is different in the current v3 baseline])[
        - Semidense projection stats and grid CNN embedding are fed into the head (candidate-specific signal).
        - Voxel validity is used as a feature (not a hard gate) in the best run config.
        - Optional trajectory encoder is enabled in the best run and carries gradient signal.
      ]
    ],
  )
]

#slide(title: [Best W&B run (v03-best)])[
  #let m = wb.metrics
  #let v = wb.vin_effective

  #grid(
    [
      #color-block(title: [Run id + checkpoint])[
        - W&B run: #code-inline[#wb.run_id]
        - Checkpoint: #code-inline[#wb.checkpoint]
        - Epoch: #m.at("epoch"), global step: #m.at("trainer/global_step")
      ]
      #color-block(title: [Key metrics (end of run)])[
        - train loss (epoch): #round(m.at("train/loss_epoch"), digits: 3)
        - val loss: #round(m.at("val/loss"), digits: 3)
        - train spearman: #round(m.at("train-aux/spearman"), digits: 3)
        - val spearman: #round(m.at("val-aux/spearman"), digits: 3)
        - train top-3 acc: #round(m.at("train-aux/top3_accuracy_epoch"), digits: 3)
        - val top-3 acc: #round(m.at("val-aux/top3_accuracy"), digits: 3)
      ]
    ],
    [
      #color-block(title: [Model config (effective)])[
        - field_dim=#v.at("field_dim"), G_pool=#v.at("global_pool_grid_size"), G_sem=#v.at("semidense_proj_grid_size")
        - semidense CNN: #v.at("semidense_cnn_enabled") (#v.at("semidense_cnn_channels") ch #sym.arrow.r #v.at("semidense_cnn_out_dim") dim)
        - trajectory encoder: #v.at("use_traj_encoder")
        - lr-AdamW (final): #m.at("lr-AdamW")
      ]
    ],
  )
]

#slide(title: [Run dynamics: v03-best])[
  #let d = wb_dyn
  #grid(
    [
      #color-block(title: [Training length and plateau])[
        - Stop criterion: #code-inline[max_epochs]=#d.max_epochs (not early stopping).
        - Loss improves early but plateaus around epoch ~#d.plateau_epoch (shallow late slopes).
      ]
      #color-block(title: [LR schedule and noise])[
        - #code-inline[lr-AdamW] decays from #d.lr_start to ~#d.lr_end.
        - Step-loss vs LR correlation (Spearman): #d.loss_lr_spearman_rho.
        - Late step-loss variability: CV ~#d.step_loss_cv_late (min #d.step_loss_min_late, max #d.step_loss_max_late).
      ]
    ],
    [
      #color-block(title: [CORAL loss trend (epoch means, imported)])[
        - train: #d.train_coral_loss_rel_random_mean_early #sym.arrow.r #d.train_coral_loss_rel_random_mean_mid #sym.arrow.r #d.train_coral_loss_rel_random_mean_late
        - val: #d.val_coral_loss_rel_random_mean_early #sym.arrow.r #d.val_coral_loss_rel_random_mean_mid #sym.arrow.r #d.val_coral_loss_rel_random_mean_late
      ]
      #color-block(title: [Interpretation])[
        - Objective remains moderately noisy; larger batch size (or grad accumulation) may reduce variance.
        - If the late-epoch slope stays near zero, schedule changes (constant LR or plateau-based decay) are a plausible next lever.
      ]
    ],
  )
]

#slide(title: [Training regime: what to try next])[
  #let d = wb_dyn
  #grid(
    [
      #color-block(title: [Batch size vs objective noise])[
        - Late-step loss CV ~#d.step_loss_cv_late suggests non-trivial stochasticity at the current batch size.
        - Hypothesis: larger batches (or grad accumulation) reduce gradient variance and improve ranking signal stability.
        - Practical test: double #code-inline[batch_size] until memory limit; scale LR if needed.
      ]
      #color-block(title: [LR schedule alternatives])[
        - Current run plateaus around epoch ~#d.plateau_epoch with a decaying LR.
        - Hypothesis: either (a) a higher early LR helps escape collapse, or (b) a plateau-aware scheduler helps late-stage refinement.
        - Practical tests:
          + constant LR (short warmup + fixed LR),
          + ReduceLROnPlateau on val loss / spearman,
          + shorter one-cycle with higher early peak (T41-style).
      ]
    ],
    [
      #color-block(title: [Keep the comparisons fair])[
        - Fix the dataset split and candidate sampling config when comparing schedules.
        - Always track: #code-inline[spearman], #code-inline[top3_accuracy], confusion matrices, and grad norms (not loss only).
      ]
      #color-block(title: [Success criteria])[
        - Confusion matrices should progressively concentrate near the diagonal (no single-bin collapse).
        - Spearman and top-3 accuracy should improve monotonically in the first few epochs.
      ]
    ],
  )
]

// #conf-matrix-sequence(
//   fig_path + "wandb/rtjvfyyp/train-figures",
//   manifest: "frames.json",
//   title: [Train confusion matrices (v03-best)],
//   caption: [Ordinal bin confusion matrices during training],
//   width: 70%,
//   caption-style: (size: 12pt, weight: "medium"),
// )

// #conf-matrix-sequence(
//   fig_path + "wandb/rtjvfyyp/val-figures",
//   manifest: "frames.json",
//   title: [Val confusion matrices (v03-best)],
//   caption: [Ordinal bin confusion matrices during validation],
//   width: 70%,
//   caption-style: (size: 12pt, weight: "medium"),
// )

// ---------------------------------------------------------------------------
// Summary + next steps
// ---------------------------------------------------------------------------

#section-slide(
  title: [Summary and Next Steps],
  subtitle: [What works, what fails, what to do next],
)

#slide(title: [Known limitations and open TODOs])[
  #grid(
    [
      #color-block(title: [Dataset constraints (ASE)])[
        - Each scene has one prerecorded egocentric trajectory (no arbitrary novel viewpoints).
        - GT meshes are available for a subset of scenes; beyond that we rely on pseudo-GT or reduced supervision.
        - Current offline cache covers only a fraction of the downloaded subset (coverage is the main scaling lever).
      ]
      #color-block(title: [Oracle pipeline / geometry tech debt])[
        - Rendering correctness is critical: avoid frame mismatches (CW90 correction) and depth convention bugs.
        - Candidate generation choices (filter vs penalize invalid views, allow backward views, roll jitter) affect the RRI distribution and CORAL bins.
      ]
    ],
    [
      #color-block(title: [Training and evaluation])[
        - Candidate ordering bias: add optional per-sample candidate shuffling in the datamodule (TODO).
        - CORAL thresholds depend on the empirical RRI distribution; changing candidate sampling changes the learning problem.
        - Validation metrics are necessary but not sufficient; we still need NBV rollout evaluation beyond per-snippet ranking.
      ]
      #color-block(title: [Practical pitfall (cached data)])[
        - For cached runs, keep #code-inline[apply_cw90_correction]=false unless cameras and reference poses are corrected in lockstep.
      ]
    ],
  )
]

#slide(title: [Key takeaways])[
  #color-block(title: [What is solid now])[
    - Oracle RRI pipeline is implemented end-to-end (candidates #sym.arrow.r renders #sym.arrow.r backprojection #sym.arrow.r RRI).
    - Offline cache enables fast training/debug loops with a typed batching contract.
    - Diagnostics are actionable (candidate validity, visibility, confusion matrices, grad norms).
  ]
  #color-block(title: [Main limitations])[
    - Data scale: current cache covers #cache.unique_scenes scenes and #cache.unique_scene_snippet unique snippets.
    - Training stability: mode collapse is still observed in some regimes; candidate-specific signal is fragile.
    - Compute: oracle labeling cost motivates more efficient labeling or more compute budget for scaling.
  ]
  #color-block(title: [Master thesis next steps])[
    - Scale dataset (more scenes, more snippets) and track coverage systematically.
    - Run a clean Optuna sweep (stationary regime; architecture toggles only).
    - Strengthen per-candidate evidence (visibility features + trajectory) and reduce collapse.
    - Add evaluation protocol for NBV rollouts (beyond per-snippet ranking).
  ]
]

#slide(title: [References])[
  #text(size: 10pt)[
    #columns(2, gutter: 0.6cm)[
      #bibliography("/references.bib", style: "/ieee.csl")
    ]
  ]
]

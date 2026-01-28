// These slides are for the final presentation of the Aria-NBV project
// Target audience: my Professor whose expertise is 3D Deep Learning in Visual Computing, he is already familiar with the project from various update meetings and our previous slides_{1,2,3}.typ, so we can skip basic explanations of concepts;
// GOAL: Presentation of the final vin-nbv architecture, findings and limitations
// The project will be continued as my Master Thesis in the upcoming semester - so we want to present all relevant information that will help us to decide on the next steps - i.e. we will need a larger dataset (more compute ressources)
// The current contents are mostly AI slopped
// We want to highlight the full OracleRRI pipeline (ase-efm dataset, candidate sampling, depth rendering, point cloud fusion, RRI computation); our offline dataset (with batching support) (how much of the available ASE subset is covered?)
// TODO: decide what other implementation details to include (i.e. we should include an overview of all metrics that are computed and logged in our lit_module - including the formulas; observations in the training dynamics...)
// Time limit for the presentation: 60 minutes (no hard limit or requirement)
// What are important findings from our Optuna sweeps and W&B runs so far?
// Symbols and important equations should be defined in shared/macros.typ and imported here so that they are consistent across the Typst paper and slides
// when refering to numeric values from the config, wandb or optuna runs they should be imported via typst's data import mechanism to avoid inconsistencies
// best wandb run ".logs/wandb/wandb/run-20260126_205313-rtjvfyyp", checkpoint "epoch=20-step=1869-train-loss=6.2684.ckpt"
// $typst-authoring should be used to gather relevant context on typst features
// $aria-nbv-context should be used to gather relevant context on the Aria-NBV project
// So far neither the slides, nor the paper contain relevant figures from wandb, our streamlit app, optuna sweeps; these are to be added in the respective sections


#import "template.typ": *
#import "notes.typ": *
#import "@preview/muchpdf:0.1.1": muchpdf
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

// Shared macros and symbols
#import "../shared/macros.typ": *
#let fig_path = "../../figures/"


#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Munich University of Applied Sciences],
  logo: [#image(fig_path + "hm-logo.svg", width: 2cm)],
  config-info(
    title: [Aria-NBV: Oracle RRI and Diagnostics],
    subtitle: [ASE + EFM3D/EVL + VIN Scoring],
    authors: [*Jan Duchscherer*],
    extra: [VCML Seminar WS24/25],
    footer: [
      #grid(
        columns: (1fr, auto, 1fr),
        align: bottom,
        align(left)[Jan Duchscherer],
        align(center)[VCML Seminar WS24/25],
        align(right)[
          #datetime.today().display("[day padding:none]. [month repr:short] [year]")
        ],
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
// ---------------------------------------------------------------------------
// Global style overrides
// ---------------------------------------------------------------------------
#set text(size: 17pt, font: "Open Sans")

#show figure.caption: set text(size: 12pt, weight: "medium", fill: theme_color_footer.darken(40%))

#show grid: set grid(columns: (1fr, 1fr), gutter: 0.8cm)

#show bibliography: set text(size: 14pt)
#show link: set text(fill: blue)
#show link: it => underline(it)

#title-slide()

// ---------------------------------------------------------------------------
// Motivation and scope
// ---------------------------------------------------------------------------

#section-slide(
  title: [Motivation and Scope],
  subtitle: [Quality-driven NBV for egocentric reconstruction],
)

#slide(title: [Why RRI for NBV?])[
  #grid(
    [
      #color-block(title: [Motivation])[
        - Coverage or information-gain proxies can miss occlusions and fine detail.
        - VIN-NBV shows that direct reconstruction-quality signals improve ranking.
        - We focus on an oracle label pipeline to make RRI supervision reliable.
      ]
      #color-block(title: [Scope in this work])[
        - Discrete candidate set around the current rig pose.
        - Offline depth rendering and point #sym.arrow mesh evaluation.
        - Diagnostics-first: validate geometry before learning a scorer.
      ]
    ],
    [
      #figure(
        image(fig_path + "VIN-NBV_diagram.png", width: 100%),
        caption: [VIN-NBV reference pipeline for quality-driven NBV. @VIN-NBV-frahm2025],
      )
    ],
  )
]

#slide(title: [Project contributions])[
  #grid(
    [
      #color-block(title: [Oracle supervision])[
        - Candidate depth rendering with metric z-buffers.
        - Candidate point clouds fused with semi-dense SLAM points.
        - Chamfer-style accuracy + completeness to compute RRI.
      ]
      #color-block(title: [Diagnostics tooling])[
        - Streamlit dashboard for candidate sampling and rendering checks.
        - RRI distributions, depth histograms, and frusta sanity checks.
      ]
    ],
    [
      #color-block(title: [Future VIN v2 scorer])[
        - EVL voxel evidence + pose encoding + semi-dense projections.
        - Ordinal CORAL head with ranking metrics.
        - Ablations driven by Optuna sweep evidence.
      ]
      #quote-block[
        This talk follows the paper: build the oracle, prove the geometry, then learn the scorer.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Dataset and oracle pipeline
// ---------------------------------------------------------------------------

#section-slide(
  title: [Data and Oracle Pipeline],
  subtitle: [ASE inputs + candidate rendering + RRI scoring],
)

#slide(title: [ASE dataset snapshot])[
  #grid(
    [
      #color-block(title: [ASE modalities])[
        - 100k indoor scenes with egocentric trajectories.
        - RGB, depth, SLAM streams + semi-dense points.
        - GT meshes for supervised subset (ASE-EFM GT split).
      ]
      #color-block(title: [Why ASE works for NBV])[
        - Calibrated rig poses + synchronized streams.
        - Mesh + semi-dense points enable oracle RRI computation.
        - EFM3D data model matches deployment expectations.
      ]
    ],
    [
      #figure(
        image(fig_path + "scene-script/ase_modalities.jpg", width: 100%),
        caption: [ASE modalities and synthetic streams. @SceneScript-avetisyan2024],
      )
    ],
  )
]

#slide(title: [ASE mesh + snippet distribution])[
  #grid(
    [
      #figure(
        image(fig_path + "gt_mesh_manhattan_sample.png", width: 100%),
        caption: [GT mesh example from ASE.],
      )
    ],
    [
      #figure(
        image(fig_path + "ase_efm_snippet_hist.png", width: 100%),
        caption: [Snippet counts per scene in the local ASE-EFM snapshot.],
      )
    ],
  )
]

#slide(title: [Oracle RRI pipeline])[
  #grid(
    [
      #color-block(title: [Stages])[
        1. Sample candidate poses around the current rig pose.
        2. Render candidate depth maps from the GT mesh.
        3. Backproject valid depths to candidate point clouds.
        4. Compute RRI via Chamfer accuracy + completeness.
      ]
      #color-block(title: [Key data products])[
        - `CandidateSamplingResult` with `PoseTW` candidates.
        - `CandidateDepths` with metric depths + validity masks.
        - `CandidatePointClouds` fused with semi-dense points.
        - `RriResult` per candidate (continuous + ordinal).
      ]
    ],
    [
      #figure(
        image(fig_path + "app/cand_frusta_kappa4_r06-29.png", width: 100%),
        caption: [Candidate frusta from the Streamlit diagnostics view.],
      )
    ],
  )
]

#slide(title: [Representative oracle label config])[
  #figure(
    kind: "table",
    supplement: [Table],
    caption: [Configuration used for Streamlit figures.],
    text(size: 8.5pt)[
      #table(
        columns: (12em, auto),
        align: (left, left),
        toprule(),
        table.header([Parameter], [Value]),
        midrule(), [Candidate count],
        [32], [Shell radius],
        [$[0.6, 2.9]$ m], [Elevation range],
        [$-15 deg$ to $25 deg$], [Azimuth spread],
        [$170 deg$], [View direction mode],
        [radial away], [View jitter caps],
        [$60 deg$ az / $30 deg$ elev], [Min distance to mesh],
        [$0.4$ m], [Depth renderer (max kept)],
        [16], [Depth z-range],
        [znear=1e-3, zfar=20], bottomrule(),
      )
    ],
  )
]

#slide(title: [Rendering + oracle scoring])[
  #grid(
    [
      #figure(
        image(fig_path + "app/candidate_renders.png", width: 100%),
        caption: [Candidate depth renders from the GT mesh.],
      )
    ],
    [
      #figure(
        image(fig_path + "app/rri_forward.png", width: 100%),
        caption: [Oracle RRI scoring diagnostics (per candidate).],
      )
    ],
  )
]

#slide(title: [Oracle RRI distribution])[
  #grid(
    [
      #color-block(title: [Skewed candidate gains])[
        - Most candidates yield marginal improvements.
        - A small fraction produce large RRI gains.
        - Example snippet (scene 81056, sample 000022):
          + median RRI = 0.0116
          + best RRI = 0.766
          + bidirectional error drops from 2.286 to 0.535
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
// VIN v2 architecture (future scorer)
// ---------------------------------------------------------------------------

#section-slide(
  title: [VIN v3 Scoring Architecture],
  subtitle: [From EVL voxel evidence to ordinal RRI prediction],
)

#slide(title: [Architecture overview])[
  #grid(
    [
      #figure(
        image(fig_path + "VIN-NBV_diagram.png", width: 100%),
        caption: [VIN-NBV baseline reference. @VIN-NBV-frahm2025],
      )
    ],
    [
      #figure(
        image(fig_path + "vin_v2/vin_v2_arch.png", width: 100%),
        caption: [Auto-generated VIN v2 module diagram (Graphviz).],
      )
    ],
  )
]

#slide(title: [View-conditioned evidence])[
  #grid(
    [
      #color-block(title: [Evidence sources])[
        - EVL voxel heads: occupancy, centerness, free-space, counts.
        - Pose encoding in rig frame (R6D + LFF).
        - Semi-dense projection stats (coverage, visibility, depth).
        - Optional frustum cross-attention over projected points.
      ]
      #color-block(title: [Reliability cues])[
        - Voxel valid fraction gates global context.
        - Projection visibility fraction tracks view reliability.
        - Optional obs-count features for point quality.
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

#slide(title: [Ordinal RRI Objective (CORAL)])[
  #grid(
    [
      #color-block(title: [RRI definition])[
        #block[
          #align(center)[
            #eqs.rri.rri
          ]
        ]
        - RRI is the relative reduction in Chamfer surface error.
        - Accuracy (P #sym.arrow M) + completeness (M #sym.arrow P).
      ]
    ],
    [
      #color-block(title: [Ordinal supervision])[
        - Bin RRI into ordered classes and predict cumulative probabilities.
        - CORAL loss enforces threshold ordering. @CORAL-cao2019
        - Report Spearman correlation and top-k bin accuracy.
      ]
      #quote-block[
        Ordinal labels keep ranking information without regressing noisy RRI values directly.
      ]
    ],
  )
]

// ---------------------------------------------------------------------------
// Diagnostics
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
    - Catch render failures (empty z-buffers, wall look-through).
    - Inspect candidate distributions before training a scorer.
  ]
]

// ---------------------------------------------------------------------------
// Optuna sweeps + W&B evidence
// ---------------------------------------------------------------------------

#section-slide(
  title: [Sweep Evidence],
  subtitle: [Optuna + W&B signals guiding architecture choices],
)

#slide(title: [Optuna sweep: regime shift])[
  #grid(
    [
      #color-block(title: [Study: vin-v2-sweep])[
        - Trials mix multiple config phases.
        - Early trials miss suggested params; later trials add scheduler + head dims.
        - Config corrections indicate runtime coercions (non-stationary evidence).
      ]
      #quote-block[
        Global importances are unreliable unless we filter by phase.
      ]
    ],
    [
      #figure(
        image(fig_path + "vin_v2/optuna_objective_vs_trial.png", width: 100%),
        caption: [Objective vs trial index (vin-v2-sweep).],
      )
    ],
  )
]

#slide(title: [Optuna toggles: evidence so far])[
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
    - Next sweep focuses on architectural toggles only.
  ]
]

#slide(title: [W&B run analysis (Jan 3, 2026)])[
  #grid(
    [
      #figure(
        kind: "table",
        supplement: [Table],
        placement: none,
        caption: [Summary of Jan 3, 2026 W&B runs with >500 steps.],
        text(size: 8.5pt)[
          #table(
            columns: (6.5em, auto),
            align: (left, left),
            toprule(),
            table.header([Run], [Summary]),
            midrule(), [Run 1],
            [Loss NaN from start; optimizer loop likely never advanced.], [Run 2],
            [Train/val losses NaN; validation Spearman 0.004; top-3 accuracy 0.191.], [Run 3],
            [Finite losses (train 10.15, val 7.69); Spearman 0.127; top-3 accuracy 0.222.], bottomrule(),
          )
        ],
      )
    ],
    [
      #figure(
        image(fig_path + "vin_v2/wandb_loss_lr_T20_vs_T30.png", width: 100%),
        caption: [W&B snapshot: train loss + LR for trials T20 vs T30.],
      )
    ],
  )
]

#conf-matrix-sequence(
  "../paper/figures/vin_v2/val-conf-mats",
  title: [VIN v2 val confusion matrices],
  caption: [Validation confusion matrices over training],
  width: 50%,
  caption-style: (size: 12pt, weight: "medium"),
)

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

#section-slide(
  title: [Summary and Next Steps],
  subtitle: [From oracle diagnostics to learned NBV policies],
)

#slide(title: [Key takeaways])[
  #color-block(title: [What we have now])[
    - Oracle RRI pipeline validated with ASE meshes and semi-dense points.
    - Streamlit diagnostics expose geometry failures early.
    - Optuna + W&B evidence guides architectural focus.
  ]
  #color-block(title: [Immediate next actions])[
    - Run a clean Optuna phase with architectural toggles only.
    - Train VIN v2 scorer on oracle labels and log ordinal metrics.
    - Extend to entity-aware weighting with EVL OBBs.
  ]
]

#slide(title: [References])[
  #text(size: 10pt)[
    #columns(2, gutter: 0.6cm)[
      #bibliography("/references.bib", style: "/ieee.csl")
    ]
  ]
]

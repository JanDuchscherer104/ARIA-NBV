// Master's thesis outlook deck for Aria-NBV.
//
// Goal: pitch a concise advisor-facing thesis agenda grounded in ideas.qmd,
// canonical project state, and the paper's discussion / extensions.

#import "template.typ": *
#import "../shared/macros.typ": *

#let fig_path = "../../figures/"
#let hestia_url = "https://johnnylu305.github.io/hestia_web"
#let gymnasium_url = "https://gymnasium.farama.org/"
#let sb3_url = "https://stable-baselines3.readthedocs.io/"
#let isaac_sim_url = "https://developer.nvidia.com/isaac/sim"
#let habitat_sim_url = "https://aihabitat.org/"
#let ai2thor_url = "https://ai2thor.allenai.org/"
#let procthor_url = "https://ai2thor.allenai.org/procthor/"

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Munich University of Applied Sciences],
  logo: [#image(fig_path + "branding/hm-logo.svg", width: 2cm)],
  config-info(
    title: [Aria-NBV: Master's Thesis Outlook],
    subtitle: [*Supervisor agenda*: _scope lock_, evidence bar, and geometry-first non-myopic planning],
    authors: [*Jan Duchscherer*],
    extra: [Master's Thesis Proposal],
    footer: [
      #grid(
        columns: (1fr, auto, 1fr),
        align: bottom,
        align(left)[Jan Duchscherer],
        align(center)[Aria-NBV Master's Thesis],
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

#set text(size: 16pt, font: "Open Sans")
#show figure.caption: set text(size: 12pt, weight: "medium", fill: theme_color_footer.darken(40%))
#show grid: set grid(columns: (1fr, 1fr), gutter: 0.8cm)
#show link: set text(fill: blue)
#show link: it => underline(it)

#title-slide()

#slide(title: [Meeting: Top 7 Items])[
  #color-block(title: [Discuss In This Order], spacing: 0.45em)[
    + *Compute?* *LRZ / workstation* first; if local, keep the stack in *WSL*.
    + *Access?* Apply now for *Aria Gen2* + *ASE simulator*.
    + *Core?* Stay on the *ASE / mesh-backed* stack and prove *geometry-first non-myopic planning*.
    + *Ablation budget?* Do the *minimum* single-step *VIN* work needed, then move to multi-step.
    + *State contract?* *Geometry-only counterfactuals* by default; *RGB / 3DGS / simulator* later.
    + *Planning target?* Pick the *return proxy*, method family, and *discrete vs continuous* phase-1 action space.
    + *Hierarchy scope?* Keep *VLA / semantic-global planning* as phase 2, not thesis core.
  ]
]

#slide(title: [Decision 1: VIN / Ablation Budget])[
  #grid(
    columns: (1fr, 1.12fr),
    gutter: 1.1em,
    [
      #block(breakable: false)[
        #color-block(title: [Topic], spacing: 0.45em)[
          + single-step *VIN* only matters if it can become a trusted *reward / critic* beyond the mesh subset
          + this decision fixes how much *#EVL / EFM backbone* and *CORAL* work is actually worth doing now
        ]
        #v(0.24cm)
        #color-block(title: [Options], spacing: 0.45em)[
          + *A. Minimum:* fix calibration, ablate *CORAL* / aux loss / key inputs, then move to multi-step.
          + *B. Architecture-first:* do heavier *VIN-v4* / backbone search before multi-step planning.
        ]
      ]
    ],
    [
      #block(breakable: false)[
        #quote-block[
          *Questions:*
          + How much *single-step VIN* work is worth doing before multi-step?
          + Do we only need *trust-building* ablations, or a stronger model that can later provide *reward / critic* estimates beyond the mesh subset?
          + Is the current *#EVL / EFM* backbone already sufficient, including a *CORAL / binning* sanity ablation?
        ]
        #v(0.22cm)
        #quote-block[
          *Recommendation:* choose *A*: do the *minimum* single-step work needed to trust the reward model, then move to multi-step
        ]
      ]
    ],
  )
]

#slide(title: [Decision 2: Counterfactual State Contract])[
  #grid(
    columns: (1fr, 1.12fr),
    gutter: 1.1em,
    [
      #block(breakable: false)[
        #color-block(title: [Topic], spacing: 0.45em)[
          + What modalities should *history* and *counterfactual* state actually use?
          + This decides whether multi-step stays geometry-first or depends on expensive RGB synthesis.
        ]
        #v(0.24cm)
        #color-block(title: [Options], spacing: 0.45em)[
          + *Geometry-only everywhere*
          + *Full history modalities* + *geometry-only counterfactuals*
          + *Counterfactual RGB / semantics* via *3DGS* or simulator
        ]
      ]
    ],
    [
      #block(breakable: false)[
        #quote-block[
          *Questions:*
          + Do we need counterfactual *SLAM PC* emulation, or is $#(symb.oracle.points) _t$ semi-dense plus $#symb.oracle.points_q$ dense already sufficient?
          + Should history use *all logged modalities*, while counterfactual state stays *geometry-only* by default?
        ]
        #v(0.22cm)
        #quote-block[
          *Recommendation:* choose *B* now; only escalate to *C* if geometry-only counterfactuals clearly fail
        ]
      ]
    ],
  )

]

#slide(title: [Decision 3: Planning Formulation])[
  #grid(
    columns: (1fr, 1.12fr),
    gutter: 1.1em,
    [
      #block(breakable: false)[
        #color-block(title: [Topic], spacing: 0.45em)[
          + phase 1 should lock the *multi-step target*, *method family*, and *action space* before worrying about online refinement
        ]
        #v(0.24cm)
        #color-block(title: [Options], spacing: 0.45em)[
          + *A. Search / discrete-shell first:* beam, MPC, close-greedy, or offline RL over the shell
          + *B. Actor-critic / continuous early:* learn continuous actions and feasibility together from the start
        ]
      ]
    ],
    [
      #block(breakable: false)[
        #quote-block[
          *Questions:*
          + What is the right *multi-step return proxy*: direct oracle #RRI sum, discounted *VIN*-predicted reward, or a mixed oracle / surrogate target?
          + Which methods fit our setting best: *beam / MPC / close-greedy*, *IQL / CQL*, or actor-critic RL?
          + Should phase 1 stay *discrete-shell*, or already include *continuous* actions?
          + Should candidate rules be handled only by masks / reward terms, or should *VIN* also predict feasibility / rule violations?
        ]
        #v(0.22cm)
        #quote-block[
          *Recommendation:* choose *A* now; lock the return proxy and method family on the discrete shell before moving toward continuous control
        ]
      ]
    ],
  )
]

#slide(title: [Decision 4: RL Regime + Data Collection])[
  #grid(
    columns: (1fr, 1.12fr),
    gutter: 1.1em,
    [
      #block(breakable: false)[
        #color-block(title: [Topic], spacing: 0.45em)[
          + online RL becomes realistic only once reward evaluation and counterfactual rendering are cheap enough
        ]
        #v(0.24cm)
        #color-block(title: [Options], spacing: 0.45em)[
          + *A. Offline first:* search, close-greedy, discrete-shell RL, and modality-contract work
          + *B. Offline + online:* add simulator-based data collection and policy refinement early
        ]
      ]
    ],
    [
      #block(breakable: false)[
        #quote-block[
          *Questions:*
          + Should we prioritize *offline only* first, or *offline + online* RL together?
          + Should early RL effort focus on *modality contract + return proxy* before simulator-based data collection?
        ]
        #v(0.22cm)
        #quote-block[
          *Recommendation:* choose *A* now; use oracle / VIN rewards offline first, and keep online RL conditional on simulator access plus a fast surrogate reward
        ]
      ]
    ],
  )
]

#slide(title: [Decision 5: Hierarchical Scope])[
  #grid(
    columns: (1fr, 1.12fr),
    gutter: 1.1em,
    [
      #block(breakable: false)[
        #color-block(title: [Topic], spacing: 0.45em)[
          + hierarchy can mean either *local geometric target-selection* or a larger *semantic-global planner*
        ]
        #v(0.24cm)
        #color-block(title: [Options], spacing: 0.45em)[
          + *A. Local hierarchy only:* target / look-at prediction plus local viewpoint realization
          + *B. Semantic-global hierarchy:* *VLA / VLM* chooses portals, entities, or frontiers for the local controller
        ]
      ]
    ],
    [
      #block(breakable: false)[
        #quote-block[
          *Questions:*
          + Should *semantic-global planning* via *VLA / VLM* be part of thesis core, or stay a phase-2 extension?
          + If hierarchy is pursued now, should it stay *local geometric* rather than full semantic-global planning?
        ]
        #v(0.22cm)
        #quote-block[
          *Recommendation:* choose *A* for thesis core; keep *VLA / semantic-global planning* explicitly out of phase 1
        ]
      ]
    ],
  )
]

#slide(title: [Theory: Geometry-First MDP Contract])[
  #grid(
    [
      #color-block(title: [Historical vs Counterfactual State], spacing: 0.45em)[
        #set text(size: 12.7pt)
        - use the *logged ego history* we already have, but keep the *implemented counterfactual view* geometry-safe
        - phase-1 state can stay *current pose + visited poses + shell candidates + valid mask*; no real RGB, #SLAM, or semantics at unvisited poses unless we synthesize them
        - rollout trees and RL episodes already instantiate this contract, but the canonical offline dataset is still *one-step*
        #set text()
        $
          #(symb.rl.s) _t = {#symb.rl.hist_ego, #(symb.oracle.points) _t, #symb.oracle.candidates}
        $
        $
          #(symb.rl.s) _t^"cf" = {#symb.rl.hist_cf, #(symb.oracle.points) _t, #symb.oracle.candidates}
        $
      ]
    ],
    [
      #color-block(title: [Action, Reward, Return Proxy], spacing: 0.45em)[
        #set text(size: 12.7pt)
        - today the *implemented* action is a *discrete shell slot*; phase 2 can factor it into target $#(symb.rl.z) _t$ plus pose $#(symb.rl.x) _t$, reusing the same env contract (#gh("aria_nbv/aria_nbv/rl/counterfactual_env.py"))
        - default reward is *oracle* #RRI, *not* VIN; VIN remains the future surrogate / critic hook
        - the current regime is *short-horizon / close-greedy*; the clean next step is to persist `top-k` chains plus discounted return into `vin_offline.counterfactuals`
        #set text()
        $
          #(symb.rl.a) _t in {0, dots, N_"shell"-1},
          quad
          q_t = "shell"[#(symb.rl.a) _t]
        $
        $
          #(symb.rl.a) _t^"phase2" = {#(symb.rl.z) _t, #(symb.rl.x) _t},
          quad
          #(symb.rl.r) _t = "RRI"(q_t)
        $
        $
          #(symb.rl.G) _t^(H) = sum_(k=0)^(H-1) gamma^k #(symb.rl.r) _(t+k), quad gamma = 0.1
        $
      ]
    ],
  )
]

#slide(title: [Theory Outlook: Reward / Q / Critic Boundary])[
  #grid(
    [
      #color-block(title: [Keywords + Equations], spacing: 0.45em)[
        #set text(size: 11.9pt)
        - *oracle #RRI reward* now; *VIN surrogate / critic* later
        - *hard masks* already handle collision, clearance, and bounds
        - *low-discount return* keeps phase 1 close-greedy
        - *Q / critic* should estimate cumulative quality, not coverage
        #set text(size: 9.6pt)
        $
          #eqs.rl.reward_geom
        $
        $
          #eqs.rl.q_backup
        $
      ]
    ],
    [
      #color-block(title: [Implemented *Now*], spacing: 0.45em)[
        #set text(size: 12.2pt)
        - candidate rules already reject *too-close*, *collision*, and *out-of-bounds* actions (#gh("aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py"))
        - the RL env already uses *oracle-RRI immediate reward* plus a flat invalid-action penalty (#gh("aria_nbv/aria_nbv/rl/counterfactual_env.py"))
        - cumulative rollout #RRI and low-$gamma$ PPO diagnostics already exist (#gh("aria_nbv/aria_nbv/pose_generation/counterfactuals.py"))
      ]
      #v(0.22cm)
      #color-block(title: [Needs Change], spacing: 0.45em)[
        #set text(size: 12.2pt)
        - make rule penalties *explicit in reward*
        - add a *VIN-backed counterfactual evaluator*
        - decide whether a *privileged critic* may use GT mesh / OBB cues
      ]
    ],
  )
]

#slide(title: [#link(hestia_url)[Hestia]: What Actually Transfers])[
  #grid(
    [
      #color-block(title: [Keep These Ideas], spacing: 0.45em)[
        - *directional observability:* encode viewing direction, not only seen / unseen
        - *hierarchical factorization:* split `_where to attend_` from `_where to move_`
        - *supervised target head:* train target / attention separately
        - *close-greedy control:* start with small gamma / short horizons
      ]
    ],
    [
      #color-block(title: [What We Should *Not* Borrow], spacing: 0.45em)[
        - do _not_ replace #RRI with coverage reward
        - move toward *target-conditioned local reads*, not fixed shell scoring forever
        - do _not_ jump straight to monolithic continuous #FiveDoF PPO
        - keep hierarchy as a *phase-2 extension*; use Hestia as *planning decomposition*
      ]
    ],
  )
]

#slide(title: [VINv4 Bridge: From Shell Scoring To Target Control])[
  #grid(
    [
      #color-block(title: [What Changes Beyond Fixed Candidate Scoring], spacing: 0.45em)[
        - predict an intermediate target latent $#(symb.rl.z) _t$ before the pose
        - supervise $#(symb.rl.z) _t$ from expected #RRI gain, uncertainty, or entity deficit
        - read local #EVL / geometry *at the target*, not only at fixed shell poses
        - bridge from discrete shell scoring toward later continuous control
        $
          #(symb.rl.a) _t^"phase2" = {#(symb.rl.z) _t, #(symb.rl.x) _t}
        $
      ]
    ],
    [
      #color-block(title: [What Stays The Same], spacing: 0.45em)[
        - keep the same quality objective: $#symb.oracle.rri$ now, $#(symb.vin.rri_hat) _t$ later
        - keep #EFM3D / #EVL geometry priors as the backbone
        - treat the target head as a *bridge*, not a replacement for the shell baseline
        - route targets through OBBs / SceneScript entities for *entity-aware* #RRI
      ]
    ],
  )
]

#slide(title: [Implemented: Rollout Scaffold])[
  #grid(
    columns: (0.88fr, 1.35fr),
    [
      #color-block(title: [What's done?], spacing: 0.45em)[
        #set text(size: 12.9pt)
        - rollouts already reuse the *same candidate-shell contract* as one-step NBV (#gh("aria_nbv/aria_nbv/pose_generation/counterfactuals.py"))
        - the oracle scorer already computes *incremental* and *cumulative* #RRI
        - horizon, branching, pruning, distance guards, and plotting already exist (#gh("aria_nbv/aria_nbv/pose_generation/plotting.py"))
        - this is a real scaffold, not a blank start
      ]
    ],
    [
      #figure(
        image(fig_path + "app/multi-step/T5K5.png", width: 100%),
        caption: [_Synthetic diagnostic_ only: multi-step counterfactual rollout tree from the implemented `app/multi-step` plotting surface.],
      )
    ],
  )
]

#slide(title: [Implemented: RL Inspector + Step Diagnostics])[
  #grid(
    columns: (0.88fr, 1.35fr),
    [
      #color-block(title: [What's done?], spacing: 0.45em)[
        #set text(size: 12.9pt)
        - the #link(gymnasium_url)[Gymnasium] RL env already exists on the discrete shell, with greedy / random baselines and #link(sb3_url)[SB3] PPO smoke coverage (#gh("aria_nbv/aria_nbv/rl/counterfactual_env.py"))
        - plotting already supports shell inspection, selected frusta, and trajectory replay (#gh("aria_nbv/aria_nbv/pose_generation/plotting.py"))
        - the RL page is evaluation-first: shell preview, episode replay, and policy comparison (#gh("aria_nbv/aria_nbv/app/panels/rl.py"))
        - greedy immediate reward already beats random; next step = evidence
      ]
    ],
    [
      #figure(
        image(fig_path + "app/multi-step/T3-greedy-rl-t3shell.png", width: 100%),
        caption: [_Synthetic diagnostic_ only: step-level shell / frusta view from the same `app/multi-step` figure family.],
      )
    ],
  )
]

#slide(title: [Roadmap: Thesis Core vs Phase-2 Extensions])[
  #grid(
    [
      #color-block(title: [VINv4 / Hierarchy Execution Path], spacing: 0.45em)[
        + directional bookkeeping on #EVL / semidense state
        + supervised target head from expected #RRI gain / uncertainty / entity deficit
        + target-aware local interpolation / feature reads
        + target-conditioned controller plus feasibility projection
        + close-greedy training before long-horizon RL
      ]
    ],
    [
      #color-block(title: [Scaling + Entity-Aware Extensions], spacing: 0.45em)[
        - scale the mesh-backed subset, anchor poses, and candidate bounds
        - compare *beam / MPC / close-greedy* against one-step greedy
        - route target selection through OBBs / SceneScript entities for *entity-aware* #RRI
        - only after that: stronger critics, counterfactual RGB / 3DGS, and semantic-global planning
      ]
    ],
  )
  #v(0.25cm)
  #color-block(title: [Desired Thesis Deliverable], spacing: 0.45em)[
    - a clear *geometry-first MDP*, a defensible *non-myopic baseline*, and an honest scope statement
  ]
]

#slide(title: [Open Questions *After* Scope Lock])[
  #grid(
    [
      #color-block(title: [Still Open Scientifically], spacing: 0.45em)[
        - how much counterfactual history should the actor observe vs critic only?
        - can a privileged critic use GT mesh / OBB / segmentation cues?
        - how do we scale beyond mesh-backed scenes?
        - which anchor poses and candidate bounds define the real training distribution?
      ]
    ],
    [
      #color-block(title: [Still Open Operationally], spacing: 0.45em)[
        - when do we broaden beyond 60 candidates and many more anchor poses?
        - how much effort should go into _VIN v4_ once the baseline exists?
        - when, if ever, do RGB / 3DGS move into thesis core?
        - what is the minimum convincing experiment set?
      ]
    ],
  )
]

#slide(title: [Backup: Simulator Options If Needed])[
  #grid(
    [
      #color-block(title: [Most Relevant Candidates], spacing: 0.45em)[
        + *ASE simulator / generation stack:* best fit if accessible
        + *#link(habitat_sim_url)[Habitat-Sim]:* best fast geometry-first sidecar
        + *#link(ai2thor_url)[AI2-THOR] / #link(procthor_url)[ProcTHOR]:* useful for semantic-global planning, not local NBV sensing
        + *#link(isaac_sim_url)[Isaac Sim]:* best public one-box multimodal candidate
      ]
    ],
    [
      #color-block(title: [Recommendation], spacing: 0.45em)[
        - this is mainly a *data-contract* question, not just a renderer choice
        - near-term, do *not* switch ecosystems before the geometry-first baseline is proven
        - if public tooling is needed: *#link(isaac_sim_url)[Isaac Sim]* main, *#link(habitat_sim_url)[Habitat-Sim]* sidecar
        - any choice still must satisfy Aria optics, RGB / #SLAM streams, GT geometry, OBB / semantics, and a plausible semi-dense / #EVL replacement
      ]
    ],
  )
]

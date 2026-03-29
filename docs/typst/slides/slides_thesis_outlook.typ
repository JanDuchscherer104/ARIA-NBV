// Master's thesis outlook deck for Aria-NBV.
//
// Goal: pitch a concise advisor-facing thesis agenda grounded in ideas.md and
// the paper's future-extensions section. Keep the deck text-first, minimal,
// and aligned with slides_4.typ.

#import "template.typ": *
#import "../shared/macros.typ": *

#let fig_path = "../../figures/"

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Munich University of Applied Sciences],
  logo: [#image(fig_path + "branding/hm-logo.svg", width: 2cm)],
  config-info(
    title: [Aria-NBV: Master's Thesis Outlook],
    subtitle: [Open questions and strongest answer candidates so far],
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

#slide(title: [Highest-priority decisions])[
  #grid(
    [
      #color-block(title: [Access now], spacing: 0.45em)[
        - workstation / cluster?
        - simulator access?
        - Aria Gen2 devkit?
      ]
    ],
    [
      #color-block(title: [Scope decision], spacing: 0.45em)[
        - stay with ASE mesh subset?
        - alternative dataset?
        - geometry-backed first?
      ]
    ],
    [
      #color-block(title: [Time budget], spacing: 0.45em)[
        - stable VIN first?
        - RL env runner first?
        - data / docs cleanup?
      ]
    ],
  )
]

#slide(title: [Ranked RL directions])[
  #grid(
    [
      #color-block(title: [Ranks 1-3], spacing: 0.45em)[
        - 1. mesh simulator + persistent memory
        - 2. MPC / beam / CEM first
        - 3. IQL / CQL on discrete candidates
      ]
    ],
    [
      #color-block(title: [Ranks 4-6], spacing: 0.45em)[
        - 4. privileged / asymmetric critic
        - 5. hierarchical / entity-aware policy
        - 6. scale anchors, candidates, geometry
      ]
    ],
    [
      #color-block(title: [Later], spacing: 0.45em)[
        - 7. LEQ / model-based offline RL
        - 8. 3DGS / egocentric splats
        - PPO / GAE only after fast rollouts
      ]
    ],
  )
]

#slide(title: [1. Simulator, state, reward])[
  #grid(
    [
      #color-block(title: [State], spacing: 0.45em)[
        - `s_t^ego`: logged modalities only
        - `s_t^cf`: logged + counterfactual history
        #set text(size: 13pt)
        #block[#align(center)[#eqs.rl.state_ego]]
        #block[#align(center)[#eqs.rl.state_cf]]
      ]
    ],
    [
      #color-block(title: [Geometry simulator], spacing: 0.45em)[
        - render from `M_GT` at queried pose
        - fuse into persistent map
        - geometry first, RGB later
        #set text(size: 13pt)
        #block[#align(center)[#eqs.rl.obs_render]]
        #block[#align(center)[#eqs.rl.memory_update]]
      ]
    ],
    [
      #color-block(title: [Geometric reward], spacing: 0.45em)[
        - eval: one-step RRI
        - train: additive log-CD improvement
        - add collision / motion cost
        #set text(size: 12.6pt)
        #block[#align(center)[#eqs.rl.reward_geom]]
      ]
    ],
  )
]

#slide(title: [2-3. Offline-only RL methods])[
  #grid(
    [
      #color-block(title: [2. Search / planning], spacing: 0.45em)[
        - beam search / MPC / CEM
        - first non-myopic baseline
        - no critic needed
        #set text(size: 13pt)
        #block[#align(center)[#eqs.rl.planner]]
      ]
    ],
    [
      #color-block(title: [3. IQL / CQL], spacing: 0.45em)[
        - discrete or local-candidate actions
        - IQL: safest first bet
        - CQL: if OOD overestimation
        #set text(size: 12.2pt)
        #block[#align(center)[#eqs.rl.q_backup]]
        #block[#align(center)[#eqs.rl.iql_q_loss]]
        #block[#align(center)[#eqs.rl.cql_loss]]
      ]
    ],
    [
      #color-block(title: [7. Later offline], spacing: 0.45em)[
        - LEQ: model-based offline RL later
        - fit lower expectile to `G_t^lambda`
        - Decision Transformer: baseline only
        #set text(size: 12.6pt)
        #block[#align(center)[#eqs.rl.return_lambda]]
        #block[#align(center)[#eqs.rl.leq_loss]]
      ]
    ],
  )
]

#slide(title: [4-5. Privileged critic, continuous policy])[
  #grid(
    [
      #color-block(title: [4. Privileged critic], spacing: 0.45em)[
        - actor: deployable state only
        - critic: GT mesh / OBB / masks
        - train-time only privilege
      ]
    ],
    [
      #color-block(title: [5. Hierarchical action], spacing: 0.45em)[
        - region / entity / look-at first
        - local pose refine second
        - easier than full 5DoF from scratch
        #set text(size: 12.8pt)
        #block[#align(center)[#eqs.rl.hier_policy]]
      ]
    ],
    [
      #color-block(title: [Continuous optimization], spacing: 0.45em)[
        - local continuous head in mesh subset
        - IQL-style actor-critic first
        - PPO / GAE only after fast rollouts
        #set text(size: 12.2pt)
        #block[#align(center)[#eqs.rl.gae]]
        #block[#align(center)[#eqs.rl.ppo_clip]]
      ]
    ],
  )
]

#slide(title: [6. Scale inside mesh subset])[
  #grid(
    [
      #color-block(title: [Generate more rollouts], spacing: 0.45em)[
        - more anchor poses
        - more than 60 candidates
        - synthetic offline trajectories
        - mixed valid / wide bounds
      ]
    ],
    [
      #color-block(title: [Counterfactual geometry], spacing: 0.45em)[
        - depth, normals, visibility
        - semantic IDs / GT depth / OBB masks
        - persistent memory over local EVL crop
      ]
    ],
    [
      #color-block(title: [Keep it trustworthy], spacing: 0.45em)[
        - no mesh decimation
        - fixed density / voxel size
        - stable bins / aux weights
      ]
    ],
  )
]

#slide(title: [8. Counterfactual RGB later])[
  #grid(
    [
      #color-block(title: [Why], spacing: 0.45em)[
        - off-trajectory RGB
        - richer backbone features
        - bridge to SLAM-like cues
      ]
    ],
    [
      #color-block(title: [Route], spacing: 0.45em)[
        - 3DGS / egocentric splats
        - per-scene or per-region fitting
        - geometry first, RGB second
      ]
    ],
    [
      #color-block(title: [Role], spacing: 0.45em)[
        - phase 2 only
        - not a blocker for RL
        - use after geometry saturates
      ]
    ],
  )
]

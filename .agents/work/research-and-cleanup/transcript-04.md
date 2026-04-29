[User-Turn01]
Please provide an overview of all issues that you previously identified, that still need to be resolved on the main branch of aria-nbv!

[Agent-Turn02]
From the earlier audit, the main unresolved problems on `aria-nbv` fall into nine clusters. I cannot verify the live `main` branch from this workspace, so this is the consolidated list of issues previously identified that still need resolution unless they have already been fixed since that audit.

## 1. Split-brain package surface: `aria_nbv.data` vs `aria_nbv.data_handling`

This is the central problem.

You appear to have two parallel implementations for the same concepts, instead of one canonical implementation plus compatibility shims. The duplicated module families previously identified were:

* `aria_nbv/data/efm_views.py` ↔ `aria_nbv/data_handling/efm_views.py`
* `aria_nbv/data/efm_dataset.py` ↔ `aria_nbv/data_handling/efm_dataset.py`
* `aria_nbv/data/efm_snippet_loader.py` ↔ `aria_nbv/data_handling/efm_snippet_loader.py`
* `aria_nbv/data/vin_oracle_types.py` ↔ `aria_nbv/data_handling/vin_oracle_types.py`
* `aria_nbv/data/vin_oracle_datasets.py` ↔ `aria_nbv/data_handling/vin_oracle_datasets.py`
* `aria_nbv/data/offline_cache.py` ↔ `aria_nbv/data_handling/oracle_cache.py`
* `aria_nbv/data/vin_snippet_cache.py` ↔ `aria_nbv/data_handling/vin_cache.py`
* `aria_nbv/data/vin_snippet_provider.py` ↔ `aria_nbv/data_handling/vin_provider.py`
* `aria_nbv/data/offline_cache_store.py` ↔ `aria_nbv/data_handling/offline_cache_store.py`
* `aria_nbv/data/mesh_cache.py` ↔ `aria_nbv/data_handling/mesh_cache.py`

Why this matters:

* duplicated logic
* ambiguous ownership
* import-path drift
* type-identity bugs
* impossible-to-trust grep results

Required fix:
pick one canonical package, then convert the other side into thin re-export shims or delete it.

## 2. Duplicate type/model definitions that should exist exactly once

These are not just cosmetic duplicates. They can break `isinstance`, typing, serialization, cache contracts, and maintenance.

Previously identified duplicates:

### `VinOracleBatch` and its collation helpers

Defined in both:

* `aria_nbv/data/vin_oracle_types.py`
* `aria_nbv/data_handling/vin_oracle_types.py`

Including duplicated helpers such as:

* `_pad_1d`
* `_pad_candidate_poses`
* `_pad_trajectory`
* `_pad_points`
* `_stack_reference_poses`
* `_stack_p3d_cameras`
* `_stack_backbone_outputs`
* `_gather_candidate`
* `_shuffle_camera_param`

These must be single-source.

### Oracle RRI cache contract types

Previously identified as duplicated across multiple locations:

* `OracleRriCacheMetadata`
* `OracleRriCacheEntry`
* `OracleRriCacheSample`

Locations called out:

* `aria_nbv/data/offline_cache_types.py`
* `aria_nbv/data_handling/oracle_cache.py`
* `aria_nbv/data_handling/cache_contracts.py`
* `aria_nbv/data_handling/cache_index.py`

### VIN snippet cache contract types

Previously identified as duplicated:

* `VinSnippetCacheMetadata`
* `VinSnippetCacheEntry`
* `VinSnippetCacheBuildResult`

Locations:

* `aria_nbv/data/vin_snippet_cache.py`
* `aria_nbv/data_handling/vin_cache.py`

### Mesh cache contract types

Previously identified as duplicated:

* `MeshProcessSpec`
* `ProcessedMesh`

Locations:

* `aria_nbv/data/mesh_cache.py`
* `aria_nbv/data_handling/mesh_cache.py`

Required fix:
move each contract/type family to one canonical module and import from there everywhere else.

## 3. Partial migration / orphaned functionality

The migration from `data` to `data_handling` was identified as incomplete.

Explicit issue previously identified:

* `SceneCoverage`
* `CacheCoverageReport`

These still live in:

* `aria_nbv/data/offline_cache_coverage.py`

but apparently do not have an equivalent in the new `data_handling` surface.

Why this matters:

* the migration is not actually finished
* parts of the old package remain semantically authoritative
* any attempt to delete old code will break hidden dependencies

Required fix:
either migrate coverage/report logic into the canonical package or explicitly declare the old package as canonical for that functionality.

## 4. Helper and utility sprawl outside shared utils/helpers

This was one of your core complaints, and it is justified.

Previously identified issues:

### `vin_snippet_utils.py` is domain-local helper sprawl

Functions called out:

* `build_vin_snippet_view`
* `vin_snippet_cache_config_hash`
* `empty_vin_snippet`

These were identified as pure helpers used by multiple modules, but living in a domain package instead of a shared utility/helper module.

### Local utility duplication

Previously noted:

* `aria_nbv/data/utils.py`
* `aria_nbv/data/vin_snippet_utils.py`

The problem is not that utilities exist. The problem is that shared helpers are scattered instead of being centralized.

Required fix:
move shared helper logic into explicit shared helper modules and leave only domain-specific orchestration in the data layer.

## 5. Plotting duplication and plotting API pollution

This was identified as one of the largest concentrations of boilerplate/slop.

Previously identified duplicate plotting helpers:

* `_to_numpy`
* `_plot_slice_grid`
* `_histogram_overlay`
* `_plot_hist_counts_mpl`
* `_pretty_label`

Previously identified module spread:

* `aria_nbv/utils/plotting.py`
* `aria_nbv/data/plotting.py`
* `aria_nbv/vin/plotting.py`
* `aria_nbv/vin/experimental/plotting.py`
* `aria_nbv/pose_generation/plotting.py`
* `aria_nbv/rendering/plotting.py`

Additional specific problem previously noted:

* `aria_nbv/vin/plotting.py` reportedly re-exports private helpers like `_to_numpy` and `_histogram_overlay` in `__all__`

Why this matters:

* duplicated plotting primitives drift subtly
* private helpers become public surface by accident
* plotting behavior becomes impossible to reason about

Required fix:
keep low-level shared plotting primitives in one canonical plotting utils module and import them everywhere else.

## 6. Duplicated `cache_dir` validator logic

Previously identified:

* `OracleRriCacheConfig`
* `VinSnippetCacheConfig`

Both were said to contain the same `@field_validator("cache_dir", mode="before")` implementation.

Why this matters:

* copy-pasted path logic drifts quietly
* config behavior stops being consistent across caches

Required fix:
extract a single shared resolver/helper or shared config mixin/base.

## 7. Dead or legacy code still present on the main branch

Explicitly identified:

* `aria_nbv/lightning/lit_module_old.py`

This is classic dead-code noise unless it is still deliberately wired in, which would itself be a problem.

Why this matters:

* confuses code search
* confuses new contributors
* encourages “just copy the old one” behavior
* makes refactoring less safe because it is unclear what is authoritative

Required fix:
delete it if dead; otherwise rename it to the canonical filename and delete the duplicate surface.

## 8. Repeated cache container dataclasses in `app/state_types.py`

Previously identified pattern:

* `DataCache`
* `CandidatesCache`
* `DepthCache`
* `PointCloudCache`
* `RriCache`

These were described as structurally identical slot dataclasses with:

* `cfg_sig: str | None = None`
* one payload field

Why this matters:

* repeated container boilerplate with no modeling benefit
* unnecessary surface area
* harder to update consistently

Required fix:
replace with one generic cache container or one shared base pattern unless there is a real semantic reason for each bespoke class.

## 9. Import-path and type-identity hazards caused by the duplication

This is the downstream effect of the above, but it is important enough to state separately.

Previously identified consequence:
old `aria_nbv.data.*` modules were still being imported by other packages like:

* `app/`
* `lightning/`
* `vin/`

while equivalent logic also existed under `data_handling/`.

Why this matters:

* `A is B` can fail for “the same” type
* `isinstance` checks can silently break
* mypy/pyright get confused
* caching/serialization/checkpoint logic can depend on accidental import paths
* maintenance becomes branchy and incoherent

Required fix:
after canonicalization, update imports repo-wide so only one implementation path is authoritative.

---

# In priority order

## Must fix first

1. `data` vs `data_handling` split-brain package duplication
2. duplicated core types/models/contracts
3. partial migration / orphaned old-package functionality
4. repo-wide import cleanup to remove mixed authority

## Next

5. helper sprawl into shared utils/helpers
6. duplicated cache-dir validator extraction
7. plotting primitive deduplication

## Then

8. delete `lit_module_old.py`
9. unify repeated session cache dataclasses

---

# The shortest honest summary

The unresolved main-branch problem is not one bug. It is an architectural state where the repo contains two competing implementations of the same data/cache/type surface, plus duplicated helpers and plotting primitives scattered across modules. Until that is collapsed into one canonical source of truth, the rest of the codebase will keep accumulating noise.

If you want, I’ll turn this into a strict cleanup checklist with one checkbox per file and per type, ready to paste into a GitHub issue.

[User-Turn03]
Please read our paper, ideas-scratchpad and literature-review to get an updated understanding of our project goals!

- paper: main.pdf (in project files and here:
- idea.qmd (scratchpad / brainstorming): https://janduchscherer104.github.io/ARIA-NBV/contents/ideas.html
- paper of interest: https://arxiv.org/html/2508.01014v3
- previous literature reviews: https://janduchscherer104.github.io/ARIA-NBV/contents/literature/

GOAL: provide a theoretically and conceptually rich literature review of the hesita paper. focus on their method, architecture, include *all* important equations and focus on all perspectives / features / methods that we could leverage for our project!

[Agent-Turn04]
I now have a more current picture of **ARIA-NBV**. Your paper frames the project as **quality-driven egocentric NBV** in ASE: you compute **oracle RRI** labels by rendering candidate depths from ground-truth meshes and scoring improvement relative to the current semi-dense SLAM reconstruction, then train a **VINv3** candidate scorer on **EVL/EFM3D voxel evidence**, pose-conditioned global context, semi-dense projection statistics, and a **CORAL** ordinal head. The current system is still explicitly a **one-step candidate-ranking** model, while your scratchpad and extensions push toward **multi-step RL**, **continuous action policies**, **entity-aware/object-centric objectives**, richer counterfactual modalities, transformer/query-centric fusion, and **directional observation encoding** for points or entities. Your literature stack is also very coherent: **VIN-NBV** for RRI, **GenNBV** for continuous RL, **EVL** for egocentric voxel lifting, and **SceneScript** for structured entity-level scene representations. ([GitHub][1])

Against that backdrop, **Hestia** is best understood as a **planning-and-control design pattern** for NBV rather than a replacement for your objective. Its core claim is that prior RL planners underperform because they (i) treat voxels too coarsely, (ii) search a hard 5-DoF continuous action space monolithically, (iii) overfit to future-return correlations, and (iv) train on insufficiently diverse data. Hestia answers with four linked ideas: a **face-aware voxel representation**, a **hierarchical look-at-then-position policy**, a **close-greedy reward design** using a very small discount factor, and **large-scale diverse training**. The paper reports at least **4% CR gain**, roughly **50% lower Chamfer Distance**, **25 FPS** inference, and real-world feasibility with a drone system. ([arXiv][2])

## My high-level reading

The strongest theoretical idea in Hestia is not “RL for NBV” by itself. It is the shift from **point-style occupancy bookkeeping** to **directional observability bookkeeping**. Their slogan, effectively, is that **occupancy alone is not enough**: a voxel can be “seen” in a coarse sense while still being poorly covered from the directions that matter for reconstruction. Hestia encodes this by attaching **six face-visibility bits** to every voxel and rewarding the discovery of newly visible faces. I would read that as a coarse **directional completeness tensor** over space. For ARIA-NBV, that is exactly the right conceptual bridge to your own note about encoding, for each point, **from which directions it has been observed**, potentially as an **incidence-angle histogram**. ([arXiv][2])

The second big idea is action factorization. Hestia does **not** predict a full 5-DoF viewpoint in one shot. It first predicts **where to look**—a 3D look-at point—and only then predicts **where to place the camera**, with yaw and pitch derived from that target. This is an extremely strong inductive bias, because it decomposes the NBV problem into **target selection** and **view realization**. In your roadmap, that maps cleanly onto **entity-aware NBV**, **object-centric views**, and eventually **continuous action policies without enumerated candidate poses**. ([CVF Open Access][3])

## Hestia’s formalism and all important equations

### 1) MDP and close-greedy objective

Hestia formulates NBV as an MDP with reward over state-action pairs and optimizes discounted return, but with a deliberately tiny discount:

[
R:\mathcal S \times \mathcal A \to \mathbb R
]

[
E_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}, \qquad \gamma=0.1
]

The unusual part is not the MDP itself; it is the choice (\gamma=0.1). That makes the objective **close-greedy**: Hestia wants immediate geometric gain and explicitly tries to avoid learning spurious long-horizon associations between a mediocre current view and a large later reward. This is theoretically important because it makes the RL objective much closer to a **receding-horizon greedy improvement problem** than to classical long-horizon exploration. ([CVF Open Access][3])

For ARIA-NBV, this matters because your current formulation is already a **greedy one-step oracle-ranking problem** based on RRI. Hestia suggests that if you move toward continuous RL, you probably should **not** jump directly to a large-(\gamma), long-horizon exploration setup. A **small-discount**, quality-driven, receding-horizon controller is much more compatible with your current oracle pipeline and with the way RRI supervision is generated. ([GitHub][4])

### 2) State: image, pose, grid, and directional face visibility

The Hestia state is

[
s_t={I_t, M_t, G_t, L_t},
]

where (I_t) is the current grayscale image, (M_t) contains the current camera state and maximum flyable height, (G_t) is a voxel grid, and (L_t) is the look-at point. The grid itself contains cumulative occupancy, positional encoding, and cumulative face visibility:

[
F_t = f_t \lor F_{t-1}.
]

That is: once a voxel face has been “observed,” it stays observed in the cumulative state. ([CVF Open Access][3])

The face visibility update is driven by the viewing direction from a voxel center (p_{v_i}) to the collision-free camera position (a'_t):

[
d_{v_i} = \frac{a'*t - p*{v_i}}{|a'*t - p*{v_i}|}.
]

For each voxel (v_i) and each of its six outward normals (n_{i,j}), Hestia marks a face as visible if the viewing direction has positive dot product with that face normal:

[
f_t(v_i,j)=\mathbf 1!\left(d_{v_i}\cdot n_{i,j}>0\right), \qquad j\in{1,\dots,6}.
]

These are the paper’s key geometry equations. They formalize the idea that **visibility is directional**, and that “covered voxel” and “well-observed geometry” are not the same thing. ([CVF Open Access][3])

For your project, this is the single most transferable representational idea. Your scratchpad already points in that direction explicitly: encode **observation direction per point**, perhaps as an **incidence-angle histogram**. Hestia provides the simplest possible instance of that idea: a six-bin orientation proxy attached to each spatial element. In ARIA-NBV, you can generalize this from axis-aligned voxel faces to **surface-normal bins**, **spherical histograms**, or **learned directional descriptors** for semi-dense points, EVL voxels, or entity tokens. ([Janduchscherer104 GitHub Pages][5])

### 3) Action space and reward

Hestia predicts only a 3-DoF position action directly:

[
\mathcal A=\left{a_t \mid a_t\in[-1,1]^3\right}.
]

Yaw and pitch are then induced by the look-at point and the collision-free adjusted position (a'_t). This is what turns a hard 5-DoF control problem into a hierarchical one. ([CVF Open Access][3])

The reward decomposes into a coverage term plus a constraint term:

[
r_t = R(s_t,a_t)=r_{\text{coverage}}(s_t,a_t)+r_{\text{constraint}}(s_t,a_t).
]

The coverage reward is

[
r_{\text{coverage}}(s_t,a_t)
============================

\frac{\sum_{i=1}^{N}\sum_{j=1}^{6}\left(F_t^{i,j}-F_{t-1}^{i,j}\right),M_{\text{col}}}{N\cdot 6}\cdot 0.3.
]

So the agent is rewarded only for the **increment in newly visible faces**, and that reward is nullified when collisions occur through the collision mask (M_{\text{col}}). The paper’s main text uses (r_{\text{constraint}}=-0.01) for unsafe or invalid actions, while the supplement clarifies that penalties cover cases like revealing no new faces, exceeding maximum height, or choosing non-free voxels. ([CVF Open Access][3])

Conceptually, this reward is much sharper than plain occupancy coverage because it is **incremental**, **directional**, and **greedy by design**. But it is still a **proxy objective**. It optimizes face coverage, not reconstruction quality itself. That is precisely where your project diverges: ARIA-NBV wants to optimize **RRI / reconstruction improvement**, not merely geometric exposure. So the transferable lesson is the **reward structure**, not the exact reward target. ([arXiv][2])

### 4) Hierarchical architecture

Hestia’s network has two stages. First, a **proposal network** consumes the voxel grid (G_t) and predicts a distribution over the look-at point (L_t). Second, a **viewpoint network** predicts the camera position (a_t) conditioned on (L_t), the current image, and pose metadata. The figure and text show the flow clearly: **state space (\rightarrow) look-at prediction (\rightarrow) multilevel grid interpolation at the target point (\rightarrow) image encoding + vector info (\rightarrow) policy head (\rightarrow) nearest collision-free projection**. ([CVF Open Access][3])

More specifically, the look-at predictor is a **3D CNN + self-attention + MLP**. The viewpoint branch uses a shallow **3D CNN grid encoder** to create a multilevel feature grid, then performs **trilinear interpolation at the proposed look-at point** to gather target-conditioned features. Those features are concatenated with a shallow **2D CNN image embedding** and vector metadata, then passed into the RL policy to predict (a_t). Finally, (a_t) is projected to the nearest collision-free point (a'_t), and the final viewpoint is ({a'_t,L_t}). ([CVF Open Access][3])

This architecture is highly relevant to ARIA-NBV because your current VINv3 already uses **pose-conditioned candidate queries**, **cross-attention over voxel tokens**, and **candidate-conditioned projection statistics**. Hestia suggests a different but complementary factorization: rather than scoring a fixed candidate set, learn an internal **target query**—the look-at point or entity of interest—and then condition a continuous viewpoint proposal on that target. That fits both your **continuous-action** ambitions and your **query-centric transformer** idea. ([GitHub][6])

### 5) Supervised targets and losses

Hestia does something very clever here: it supervises the look-at predictor directly using the centroid of the **currently uncaptured ground-truth surface**. The target is

[
L_t^{\text{gt}}
===============

\frac{\sum_{v_i\in U} w_{v_i},p_{v_i}}
{\sum_{v_i\in U} w_{v_i}},
]

where (U) is the set of voxels containing uncaptured faces, and the voxel weight is simply the number of uncaptured ground-truth faces in that voxel:

[
w_{v_i} = \sum_{f\in F^{\text{gt}}_{v_i}} 1.
]

The proposal loss is then

[
L_{\text{proposal}}=|L_t-L_t^{\text{gt}}|^2.
]

The viewpoint network uses the RL loss plus an auxiliary feasibility loss

[
L_{\text{aux}}=|a_t-a'_t|^2,
]

and the total training objective is

[
L_{\text{all}} = L_{\text{RL}} + 0.5,L_{\text{aux}} + L_{\text{proposal}}.
]

The authors also note that they deliberately **detach** the gradient flow between the networks rather than relying on pure joint RL training for both stages. ([CVF Open Access][3])

This is extremely relevant for ARIA-NBV. Your scratchpad asks how to combine **imitation-style supervision** from oracle RRI with a **continuous action policy** that does not take enumerated candidate encodings as input. Hestia gives one principled answer: supervise an intermediate **target variable** directly from geometry, then let RL optimize the final motion conditioned on that target. For you, that target could be a **look-at point**, an **entity center**, an **uncertainty peak**, or an **RRI-gradient proxy**. ([Janduchscherer104 GitHub Pages][5])

### 6) Training setup and capacity

Hestia is lightweight by current standards: around **4.9M parameters**, trained with **PPO** in **stable-baselines3**, **5M iterations**, a **20³ grid**, parallel simulation of **256 scenes**, about **24 hours on an RTX A6000**, and episodes ending at **50 captures** or **0.9 face coverage**. The main experiments use processed **Objaverse** for broad training and **Houses3K** for fair comparison. ([ar5iv][7])

That matters because it shows the paper is not just an elegant abstraction; it is a fairly practical control stack. But for ARIA-NBV the most interesting implication is different: the gains likely come less from raw model size and more from **better state geometry**, **action decomposition**, and **training distribution design**. That is good news for you, because your EVL-backed system already has a much stronger perceptual prior than Hestia’s grayscale+grid setup. ([ar5iv][7])

## What Hestia contributes conceptually

The paper’s deepest contribution is that it turns NBV from “predict the next camera pose” into a three-part problem:

1. **What region / entity / surface deficit deserves attention now?**
2. **From which direction would that deficit best be resolved?**
3. **What feasible camera motion realizes that observation under constraints?** ([CVF Open Access][3])

Seen that way, Hestia is not just another RL planner. It is a **factorized observability model**. The look-at point is a latent variable for **attention allocation**. The face-visibility tensor is a coarse model of **directional epistemic incompleteness**. The collision-free projection is a **feasibility operator**. And the low discount factor makes the whole system approximate a greedy improvement process with structured control. ([arXiv][2])

That is exactly why it is interesting for your project. ARIA-NBV already has the stronger reconstruction objective: **RRI** rather than face coverage. What Hestia adds is a better story for **continuous control**, **directional observation encoding**, and **intermediate supervision**. In other words: keep your **quality objective**, borrow their **planning decomposition**. ([Janduchscherer104 GitHub Pages][8])

## What I would leverage for ARIA-NBV

### 1) Add directional observability to your scene representation

This is the cleanest import. Your own notes already propose encoding from which directions points were observed, potentially via incidence-angle histograms. Hestia shows a minimal discrete version of that idea with six face bins per voxel. For ARIA-NBV, I would attach a **directional observation histogram** to each semi-dense point, pooled voxel, or entity token, then expose summary statistics of that directional coverage to the RRI predictor. This could be done as per-point spherical bins, normal-aligned bins, or learned directional embeddings. ([Janduchscherer104 GitHub Pages][5])

### 2) Replace “score only fixed candidates” with a target-conditioned continuous controller

Your current VINv3 is a candidate scorer, not yet a continuous policy. Hestia suggests a principled bridge: predict an intermediate **look-at / target point** from the scene state, then predict a relative camera translation conditioned on that target. That maps directly onto your scratchpad question about combining oracle-RRI imitation with a continuous policy that does not consume candidate encodings. ([GitHub][6])

### 3) Define target supervision from missing quality, not just missing coverage

Hestia’s (L_t^{gt}) is a weighted centroid of **uncaptured faces**. In ARIA-NBV, the analogous target need not be coverage-based. It could be the centroid of **largest expected RRI gain**, **largest completeness deficit**, **highest uncertainty**, or even **entity-specific missing surface**. Your extensions section already defines an entity-aware objective with weighted scene and entity terms; Hestia gives you a concrete mechanism for turning that into a supervised **target proposal branch**. ([CVF Open Access][3])

### 4) Add feasibility projection rather than only penalties

Hestia does not merely penalize invalid actions; it **projects** the predicted camera position to the nearest collision-free point. Your current oracle pipeline already has explicit collision and free-space pruning for sampled candidates. If you move to continuous actions, a learned controller should probably output an unconstrained proposal and then pass it through a **feasibility projector** based on mesh, occupancy, or EVL free-space evidence, rather than hoping penalties alone will produce valid motions. ([GitHub][9])

### 5) Make multi-scale interpolation target-aware

Hestia’s multilevel trilinear interpolation at the look-at point is a strong design. Your VINv3 already does pose-conditioned global pooling plus semi-dense projection statistics, but Hestia suggests adding **explicit target-conditioned local feature reads** from the voxel grid. In EVL terms, that means sampling features not only globally or through candidate frusta, but also at predicted **target points**, **entity centers**, or **surface-deficit peaks**. ([CVF Open Access][3])

### 6) Use close-greedy RL rather than fully long-horizon RL at first

Your paper is explicit that current scope is one-step ranking and that full next-view policy learning remains future work. Hestia’s (\gamma=0.1) is a useful warning: for NBV, especially when the quality signal is already expensive and somewhat myopic, long-horizon RL can create bad credit assignment. A receding-horizon or low-discount controller that optimizes a learned **RRI surrogate** is a much cleaner first step than a high-discount end-to-end policy. ([GitHub][4])

### 7) Scale supervision via diversity, not only model capacity

Hestia’s data story matters. The paper argues that diverse large-scale training materially improves robustness. Your scratchpad says your oracle RRI pipeline currently depends on the subset of ASE scenes with GT meshes, that you have only used a fraction of available snippets, that you can sample far more than 60 candidates, and that you should vary the anchor pose instead of always using the last frame. Hestia strongly reinforces that direction: widen the distribution of scenes, anchors, candidate generation parameters, and positional shifts before assuming architecture is the main bottleneck. ([arXiv][2])

### 8) Use Hestia’s hierarchy in an entity-centric version

Because your roadmap explicitly mentions entity-aware NBV and EVL already predicts OBBs, Hestia’s hierarchy can be lifted almost directly into an **entity-centric controller**: first choose the target entity or target point inside an OBB; then choose the viewpoint that best resolves missing geometry for that entity; finally project to a feasible egocentric pose. If you later compare **EVL** with **SceneScript**, the same hierarchy could operate over either **voxel targets** or **entity tokens / structured scene commands**. ([GitHub][10])

## What I would *not* copy blindly

I would not replace your quality objective with Hestia’s face-coverage reward. Hestia is still fundamentally **coverage-oriented**, even if it is a much smarter coverage proxy than raw occupancy. Your whole project is motivated by the fact that coverage can diverge from actual reconstruction quality, especially in cluttered indoor scenes with occlusions and fine detail. That argument remains correct. ([GitHub][1])

I also would not keep Hestia’s directional encoding at the exact six-face axis-aligned level. That is elegant and cheap, but probably too coarse for your setting. Egocentric indoor scenes, oblique surfaces, and semantically meaningful object parts call for richer directional descriptors—at least **normal-aligned bins** or **incidence histograms**, and possibly learned directional tokens. Your own notes already point beyond Hestia’s discretization. ([CVF Open Access][3])

Finally, I would not throw away your foundation-model priors. Hestia uses a lightweight grayscale+grid policy trained from scratch; ARIA-NBV already has access to much richer egocentric priors through **EVL/EFM3D**, and your roadmap also considers **SceneScript** as a more explicit semantic representation. So the right synthesis is not “replace EVL with Hestia,” but “use Hestia’s control decomposition on top of EVL or SceneScript representations.” ([ar5iv][7])

## Bottom line

My strongest takeaway is this: **Hestia is most valuable to ARIA-NBV as a blueprint for continuous, target-conditioned planning under directional observability—not as a replacement for RRI.** Keep your **oracle quality supervision**, **EVL-backed scene representation**, and **entity-aware extension path**. Import from Hestia the **directional observation encoding**, **look-at / target proposal stage**, **feasibility projection**, **multiscale target-conditioned feature reads**, and **close-greedy training philosophy**. That hybrid would be much closer to your stated project trajectory than either plain VIN-style candidate ranking or plain coverage-based RL alone. ([GitHub][1])

The concrete next step I would recommend is: design a **VINv4 / policy hybrid** with three new ingredients only—**directional observation histograms**, a supervised **target-point proposal head**, and a **continuous relative-translation policy with feasibility projection**—while keeping the downstream target as **oracle RRI or entity-aware RRI**. That would be the most faithful and highest-signal way to operationalize Hestia inside ARIA-NBV. ([Janduchscherer104 GitHub Pages][5])

[1]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/main.typ "https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/main.typ"
[2]: https://arxiv.org/html/2508.01014v3 "https://arxiv.org/html/2508.01014v3"
[3]: https://openaccess.thecvf.com/content/WACV2026/papers/Lu_Hestia_Voxel-Face-Aware_Hierarchical_Next-Best-View_Acquisition_for_Efficient_3D_Reconstruction_WACV_2026_paper.pdf "https://openaccess.thecvf.com/content/WACV2026/papers/Lu_Hestia_Voxel-Face-Aware_Hierarchical_Next-Best-View_Acquisition_for_Efficient_3D_Reconstruction_WACV_2026_paper.pdf"
[4]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/03-problem-formulation.typ "https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/03-problem-formulation.typ"
[5]: https://janduchscherer104.github.io/ARIA-NBV/contents/ideas.html "https://janduchscherer104.github.io/ARIA-NBV/contents/ideas.html"
[6]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/06-architecture.typ "https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/06-architecture.typ"
[7]: https://ar5iv.labs.arxiv.org/pdf/2508.01014 "https://ar5iv.labs.arxiv.org/pdf/2508.01014"
[8]: https://janduchscherer104.github.io/ARIA-NBV/contents/literature/vin_nbv.html "https://janduchscherer104.github.io/ARIA-NBV/contents/literature/vin_nbv.html"
[9]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/05-oracle-rri.typ "https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/05-oracle-rri.typ"
[10]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/10a-extensions.typ "https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/10a-extensions.typ"

[User-Turn05]
Use VLM action model for global planning. We can also give it descriptors of the pointcloud.
So the llm is easily be able to se doorways and how to cross them. Or to identify an object we care about.

[Agent-Turn06]
Yes. This is a strong direction, but only if you keep the decomposition clean:

**VLM/VLA for global semantic planning.
Geometric NBV controller for local metric execution.**

That is the right division of labor for ARIA-NBV.

## Core idea

Your current stack is strong on **local geometric value estimation**: where a view should improve reconstruction quality. What it still lacks is a **semantic long-horizon planner** that can decide things like:

* “the target object is probably in the next room”
* “the unexplored region behind that doorway is high-value”
* “this portal should be crossed now”
* “this cabinet / shelf / table deserves close inspection”

A VLM action model can fill exactly that gap.

But I would **not** let it output raw continuous poses. That is where these models become unreliable.
Instead, let it output **grounded subgoals over map elements**.

For example:

```json
{
  "subgoal_type": "cross_portal",
  "portal_id": "doorway_7",
  "target_region": "kitchen",
  "reason": "large unexplored connected free-space behind portal; likely object category match"
}
```

or

```json
{
  "subgoal_type": "inspect_entity",
  "entity_id": "cabinet_12",
  "view_hint": "front-left-close",
  "reason": "object-of-interest likely inside or on top"
}
```

Then your local controller turns that into actual motion and viewpoints.

## Why this fits your project unusually well

ARIA-NBV is already moving toward:

* **entity-aware objectives**
* **SceneScript / structured scene abstraction**
* **continuous policy extensions**
* **directional observability**
* **semantic relevance weighting**

A VLM planner naturally sits **above** those components.

The clean hierarchy is:

[
\pi_{\text{global}}(g_k \mid S_k^{\text{semantic}}) \to \text{subgoal } g_k
]

[
\pi_{\text{local}}(a_t \mid x_t, g_k) \to \text{metric motion/view}
]

where:

* (S_k^{\text{semantic}}): semantic/topological state for long-horizon reasoning
* (g_k): grounded subgoal
* (x_t): local geometric state for control and NBV
* (a_t): local action / pose update

This is essentially a **semantic POMDP over regions, portals, and entities**, with your current NBV model acting as the local information-gathering controller.

## What the VLM should actually see

Do **not** feed it raw point clouds alone and hope for the best.
Give it a **grounded multimodal state**.

The best input is probably a combination of:

### 1. Egocentric visual evidence

A few current or recent RGB views, because VLMs are naturally strong there.

### 2. Structured pointcloud descriptors

Not the full cloud. A distilled description of the geometry and semantics.

### 3. Topological graph

Rooms, portals, explored components, frontier regions.

### 4. Entity memory

Objects, OBBs, descriptors, uncertainty, task relevance.

### 5. Mission/query

“What are we looking for?” or “what matters most?”

So the VLM input should look more like **scene memory + task + supporting views**, not just “here is a giant pointcloud”.

## The right pointcloud descriptors

This is the key design problem. The descriptors should expose the parts of geometry that are useful for reasoning.

I would split them into five token families.

### A. Portal / doorway tokens

These are probably the most important for cross-room planning.

Each portal token could contain:

* `portal_id`
* center / normal / width / height
* connected free-space components
* clearance / traversability
* explored ratio on the far side
* uncertainty behind portal
* visibility evidence through portal
* semantic hints from nearby images
* estimated room-transition prior

This is what allows the VLM to reason:
“doorway_3 likely leads to a bathroom / kitchen / bedroom; crossing it is worthwhile.”

### B. Frontier tokens

These represent unexplored-but-reachable regions.

Each frontier token could include:

* location
* free-space volume behind it
* distance cost
* expected visibility gain
* expected semantic novelty
* estimated object likelihood
* reconstruction uncertainty deficit

This turns generic frontier exploration into **task-aware frontier selection**.

### C. Entity tokens

For object-centric planning.

Each entity token could include:

* `entity_id`
* class / open-vocabulary caption
* confidence
* OBB / centroid / extent
* visibility completeness
* directional observation coverage
* occlusion level
* task relevance score
* likely support relation or container relation

This is where SceneScript-style abstractions become especially valuable.

### D. Region / room tokens

For long-horizon structure.

Each room/region token could include:

* region ID
* semantic type guess
* explored fraction
* connectivity
* object priors
* clutter level
* reconstruction quality summary
* travel cost from current pose

### E. Local geometry summary tokens

Compact summaries for the current neighborhood:

* free-space shape
* narrow passage flags
* stair / step / slope hazard
* visibility bottlenecks
* surface-normal / directional-observation histogram
* near-field occlusion structure

## Doorways are exactly the right use case

Your intuition about doorways is good.

Doorways are not just geometric openings. They are **topological bottlenecks** and **semantic transitions**.

A purely local NBV controller often struggles with them because the immediate local gain can look ambiguous:

* the current room may still have residual local value
* the next room is only partially visible
* long-horizon reward is hard to estimate geometrically

A VLM can use semantic and commonsense priors to reason:

* crossing a doorway often reveals a new room-scale component
* a requested object category may be more likely beyond that portal
* the visual hint beyond the portal may suggest a useful transition
* some rooms are semantically richer for the current goal than others

So yes: **portal selection** is one of the cleanest, highest-signal tasks for a VLM planner.

But again, the VLM should select **which** doorway to cross, not the exact body trajectory through it.

## The architecture I would recommend

### Level 1: World abstraction

Build a structured memory from EVL / pointcloud / SLAM:

* topological graph of regions and portals
* entity graph
* uncertainty / quality map
* directional observation summaries
* task-conditioned relevance scores

This can be text, JSON, or tokenized embeddings.

### Level 2: Global VLM planner

Input:

* task/query
* recent images
* current map summary
* portal/entity/frontier descriptors

Output:

* one grounded subgoal from a restricted action vocabulary

For example:

* `cross_portal(portal_id)`
* `inspect_entity(entity_id, view_hint)`
* `approach_frontier(frontier_id)`
* `reobserve_region(region_id, deficit="backside coverage")`
* `search_for_object(category="fire extinguisher")`

### Level 3: Local geometric executor

This is your domain.

Given a subgoal, your local planner chooses:

* feasible path
* local viewpoints
* NBV sequence
* collision-safe poses
* final inspection view

This layer should still optimize:

* RRI / quality improvement
* visibility of the subgoal entity/region
* safety / traversability / time cost

### Level 4: Verifier / replanner

After each executed chunk:

* update scene memory
* verify whether the subgoal was satisfied
* call the VLM again only when needed

That keeps planning stable and reduces hallucinated drift.

## How this connects to Hestia

Hestia gives you a nice conceptual bridge.

Their hierarchy is:

1. choose where to look
2. choose where to place the camera

Your extension becomes:

1. choose **what semantic target / portal / region / object** matters globally
2. choose **what local target point or entity view** matters geometrically
3. choose **what feasible camera motion** realizes it

So your hierarchy becomes richer than Hestia:

[
\text{semantic target selection}
\rightarrow
\text{geometric target proposal}
\rightarrow
\text{local viewpoint control}
]

That is a very natural next step for ARIA-NBV.

## The most important design constraint

Do not make the VLM reason over unconstrained free-form text alone.

Force it to select from **grounded IDs**:

* `portal_3`
* `frontier_5`
* `entity_12`
* `region_2`

and force it to justify selection using provided descriptors.

That gives you:

* interpretability
* debuggability
* feasibility checks
* lower hallucination risk
* easier evaluation

The planner should be more like a **semantic policy over grounded symbols** than a chatty assistant.

## What I would be skeptical about

There are three main failure modes.

### 1. Hallucinated geometry

The VLM may infer structure that is not actually present.

Fix:

* never let it invent geometry
* only allow choices among existing grounded candidates
* add a verifier before execution

### 2. Weak metric grounding

VLMs are bad at exact spatial control.

Fix:

* use them only for subgoal selection
* keep metric execution in the local planner

### 3. Descriptor overload

If you dump a huge pointcloud summary into the model, signal will vanish again.

Fix:

* curate a compact, typed memory
* only surface the most decision-relevant candidates
* use retrieval: show the top (K) portals / frontiers / entities, not all of them

## The most promising research angle

The strongest version of this idea is not just “VLM for exploration.”

It is:

**language-conditioned semantic-global planning over a structured egocentric 3D memory, with reconstruction-quality-aware local execution.**

That is much sharper.

It means the system can answer goals like:

* “find the coffee machine”
* “finish the backside coverage of the chair and table cluster”
* “cross into the adjacent room with the highest expected task relevance”
* “inspect containers likely to occlude the target object”

This is much closer to a real embodied intelligence system than pure candidate-ranking NBV.

## The representation I would actually prototype first

I would start with a very small grounded action space:

* `CROSS_PORTAL(portal_id)`
* `INSPECT_ENTITY(entity_id)`
* `APPROACH_FRONTIER(frontier_id)`
* `COMPLETE_ENTITY(entity_id)`
* `COMPLETE_REGION(region_id)`

And a compact descriptor schema like:

```json
{
  "current_region": "hallway_1",
  "task": "find the fire extinguisher",
  "portals": [
    {
      "id": "portal_2",
      "connects_to": "unknown_region_a",
      "width_m": 0.86,
      "distance_m": 1.8,
      "free_space_beyond": 12.4,
      "semantic_hint": "kitchen-like surfaces visible",
      "uncertainty_beyond": 0.78
    }
  ],
  "entities": [
    {
      "id": "entity_9",
      "label": "cabinet",
      "region": "unknown_region_a",
      "confidence": 0.74,
      "task_relevance": 0.55,
      "visibility_completeness": 0.31
    }
  ],
  "frontiers": [
    {
      "id": "frontier_4",
      "distance_m": 2.1,
      "expected_quality_gain": 0.43,
      "semantic_novelty": 0.67
    }
  ]
}
```

That is already enough for useful high-level reasoning.

## My blunt recommendation

Yes, use a VLM action model for **global planning**.
No, do not use it as the direct viewpoint generator.

Use it to choose:

* which doorway to cross
* which room to enter
* which object/entity to inspect
* which frontier is semantically worthwhile
* which region has the highest task-conditioned value

Then let your geometric planner do what it is good at:

* safe motion
* local NBV
* quality-aware viewpoint choice
* collision handling
* metric refinement

That decomposition is clean, realistic, and much more likely to work than asking a VLM to directly solve metric NBV.

The most valuable next step is to formalize a **grounded semantic action space plus pointcloud descriptor schema** for portals, frontiers, entities, and regions.

[User-Turn07]
What would be simulators that could fit well into our existing aria nbv ecosystem system? So that we could do online with all modalities for counterfactual views.

Get all (necessary) modalities from our repo and paper.

[Agent-Turn08]
Based on your paper and docs, the simulator question is really a **data-contract** question. ARIA-NBV does **not** just need RGB-D. It needs a simulator that can either natively emit, or let you derive, the exact sensory and geometric products your current pipeline consumes. ([GitHub][1])

The **online modality set** I would treat as necessary is:

* **Aria-like multi-camera image streams**: RGB plus the SLAM streams EVL expects (`rgb`, `slaml`, `slamr`), with synchronized per-frame intrinsics/extrinsics and rig/world poses. Your paper says ASE snippets provide synchronized RGB + SLAM camera streams, inertial signals, rig trajectory, and calibration, and your EFM3D notes make the EVL input contract explicit as RGB + SLAM-L + SLAM-R + camera calibration + semi-dense points. ([GitHub][1])
* **Aria optics fidelity**: this matters more than it looks. Project Aria’s RGB and SLAM cameras use the **FisheyeRadTanThinPrism** model, not a generic pinhole camera. If the simulator cannot model that, you will need a custom distortion layer on top. ([Facebook Research][2])
* **Counterfactual geometric supervision**: depth / ray distance, instance or semantic segmentation, GT mesh, GT OBBs / categories, and the ASE scene-language / floor-plan structure. Those are already part of the ASE/ARIA-NBV ecosystem and are exactly the signals your notes call out for synthesizing missing counterfactual modalities. ([Facebook Research][3])
* **SLAM-like structural products**: semi-dense points, semi-dense observations / visibility tables, inverse-distance uncertainty, and observation counts. Those are central to your current VINv3 / EVL contract and are the hard part to reproduce online. ([GitHub][1])
* **Derived 3D fields for planning**: EVL-style voxel evidence such as `occ_input` / free-space, projection counts, occupancy / centerness heads, plus the extra mesh-derived signals your notes mention for counterfactual poses: SDF, normals, directional visibility, collision / free-space checks, and projected semi-dense statistics. ([GitHub][4])

So the real bottleneck is **not** “can the simulator render RGB and depth?” The real bottleneck is: **can it support Aria optics, multi-camera synchronization, and a plausible online replacement for MPS semi-dense products + EVL voxel evidence?** ([Facebook Research][2])

My ranking would be this.

**1. Best overall if you can get access: the actual ASE simulator / Meta internal ASE generation stack.**
This is the cleanest fit because ASE is already simulated with Aria sensor characteristics, and the public data format already matches the things you care about: RGB fisheye, ray-depth, instance maps, scene language, ground-truth trajectories, and MPS-format semi-dense points and observations. Your own notes explicitly call out “apply for … the simulator that was used for generating the ASE dataset.” The catch is that the public docs I found document the **dataset and tools**, not a public simulator release, so this is the best option only if you can obtain access through the Aria / ASE research channel. ([Facebook Research][5])

**2. Best public “one-box” option: NVIDIA Isaac Sim.**
If you want one public platform that can get closest to your full multimodal online stack, Isaac Sim is the strongest candidate. Its Replicator stack supports **online generation directly into memory** for training; it can collect data from **multiple cameras** at chosen simulation timepoints; camera outputs include RGB, depth, point clouds, semantic labels, instance labels, and 2D/3D bounding boxes; camera annotators also expose normals and motion vectors; and Isaac has simulated **IMU** and **RTX lidar** support. Most importantly for Aria alignment, its camera API exposes **fisheye polynomial** properties and **Radial-Tangential Thin Prism** distortion parameters, which is much closer to Aria’s documented camera family than generic pinhole rendering. ([Isaac Sim Documentation][6])

The downside of Isaac is not rendering capability; it is **ecosystem impedance**. You would still need to build your own **Aria rig definition**, your own **ATEK/EFM export shim**, and your own **MPS-like semi-dense / observation-table pipeline** on top of Isaac’s raw sensor outputs. In other words: Isaac can give you the raw materials, but not your current ARIA-NBV products out of the box. That is an inference from the fact that Isaac’s documented outputs are camera / annotator / sensor data, while your stack consumes MPS-style semi-dense products and EVL-specific voxel evidence. ([Isaac Sim Documentation][7])

**3. Best fast geometry-first research option: Habitat-Sim.**
Habitat-Sim is a very good fit if your immediate goal is **high-throughput counterfactual rendering and RL prototyping**, not perfect Aria realism. Its documented sensor API exposes **COLOR, DEPTH, and SEMANTIC** sensors; it supports **fisheye** sensor subtypes; it has a semantic scene graph with **regions / objects / semantic scene**; and its simulator API lets you render all registered sensors in batch and use pathfinding / navmesh utilities. That makes it a strong option for millions of candidate views, doorway reasoning, region transitions, and geometry-heavy policy training. ([AI Habitat][8])

But Habitat is still a step away from your exact ecosystem. The documented fisheye model is **double-sphere**, while Aria uses **FisheyeRadTanThinPrism**, and the official Habitat docs emphasize image / semantic sensors and scene graphs rather than IMU or MPS-like SLAM products. So Habitat is excellent for **fast view generation and topology-aware planning**, but you would need more custom work than in Isaac to match Aria sensor fidelity and your semi-dense data contract. The “more custom work” part is my inference from the published APIs. ([AI Habitat][9])

**4. Best if you want extra geometric modalities like normals / flow: iGibson.**
iGibson officially supports RGB, surface normals, segmentation, 3D point clouds, depth maps, optical flow, scene flow, and LiDAR. That is actually very attractive for your “counterfactual feature synthesis” questions, because normals, flow, and dense point clouds can be useful auxiliaries for local planning or learning view synthesis modules. ([Stanford Vision Lab][10])

Still, iGibson is a weaker ecosystem fit than Isaac or Habitat for ARIA-NBV. It is not Aria-native, there is no documented bridge to ATEK / EFM3D / MPS products, and you would still need to create the Aria camera rig and semi-dense export layer yourself. So I would treat it as a good **research sandbox for auxiliary modalities**, not the main simulator around which to standardize the project. ([Stanford Vision Lab][10])

**5. Best for semantic/global-planning experiments, but not for the core local counterfactual renderer: AI2-THOR / ProcTHOR.**
AI2-THOR exposes RGB frames, depth frames, instance segmentation frames, per-object masks, 2D detections, object metadata, and multiple third-party camera streams. That makes it genuinely useful for the kind of **VLM-driven global planning** you’ve been discussing: doorways, room transitions, object search, semantic subgoal selection. ([AI2-THOR][11])

But I would not use AI2-THOR as the main ARIA-NBV counterfactual sensor engine. It is much weaker on Aria camera fidelity, semi-dense SLAM realism, and the exact geometric supervision products your current RRI / EVL stack expects. It is useful above the local controller, not as the base sensor simulator. ([AI2-THOR][11])

My blunt recommendation is:

**Near-term, do not switch ecosystems unless you have to.**
The highest-signal path is to build an **ASE-native counterfactual sensor server** first. You already have ASE scene language, GT meshes, depth, instances, GT trajectories, object mappings, and MPS-format semi-dense products in the same ecosystem. Your own notes already suggest generating counterfactual RGB with splats / world models and deriving mesh-based signals such as SDF, normals, directional visibility, GT OBBs, and GT segmentation for counterfactual poses. So the most natural next step is: keep ASE assets, keep Project Aria optics, and add an online sensor layer that can query novel poses and then regenerate the derived products your VIN / EVL stack needs. ([Facebook Research][3])

If you need a **public simulator** now, I would choose **Isaac Sim** as the main candidate and **Habitat-Sim** as the fast sidecar. Isaac is the best public fit for the full multimodal sensor stack; Habitat is the best lightweight engine for large-scale counterfactual rendering and RL loops. ProcTHOR / AI2-THOR is the one I would reserve for **global semantic planning** experiments rather than for local NBV sensor generation. ([Isaac Sim Documentation][6])

If you want, I can turn this into a **modality-by-simulator matrix** with columns like `rgb/slam cams`, `Aria optics`, `depth`, `instance/semantic`, `OBBs`, `IMU`, `scene graph`, `MPS-like semidense`, `ATEK export effort`, and then give you a concrete recommendation for **Phase 1 / Phase 2 / Phase 3** integration.

[1]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/04-dataset.typ "raw.githubusercontent.com"
[2]: https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models?utm_source=chatgpt.com "Camera Intrinsic Models | Aria Gen 1 Docs"
[3]: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format "https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format"
[4]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/typst/paper/sections/06-architecture.typ "raw.githubusercontent.com"
[5]: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset "Aria Synthetic Environments Dataset | Aria Gen 1 Docs"
[6]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_online_generation.html "Online Generation — Isaac Sim Documentation"
[7]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html"
[8]: https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.html "habitat_sim.sensor | Habitat Sim Docs"
[9]: https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.FisheyeSensorDoubleSphereSpec.html "https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.FisheyeSensorDoubleSphereSpec.html"
[10]: https://stanfordvl.github.io/iGibson/renderer.html "Renderer — iGibson 2.2.2 documentation"
[11]: https://ai2thor.allenai.org/robothor/documentation/?utm_source=chatgpt.com "RoboTHOR API Reference"
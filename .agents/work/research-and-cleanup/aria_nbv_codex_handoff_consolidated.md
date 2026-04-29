# ARIA-NBV consolidated Codex handoff

**Purpose.** This document merges the information from the four GPT-5.5 Pro transcript files into one Codex-ready planning handoff. It is intentionally written as a source-of-truth briefing for entering Codex **plan mode**: project state, decisions, architecture, methods, literature, repo cleanup, simulator options, and actionable next steps.

**Input transcripts merged.**

- `transcript-01.md`: multi-step counterfactual rollouts, RL literature, Fisher/3DGS NBV, thesis direction, broad research, online simulator discussion, GitHub roadmap, ruthless simplification, and Rerun diagnostics.
- `transcript-02.md`: scaffold comparison with `prml-vslam`, issue roadmap, and docs/package slop audit.
- `transcript-03.md`: scaffold comparison, issue roadmap, ruthless simplification, and package decluttering audit.
- `transcript-04.md`: unresolved main-branch issues, Hestia literature review, VLM/global planner design, and simulator/modality requirements.

**Important caveat for Codex.** The repo-state observations below are merged from the transcripts, not freshly re-audited against the live `main` branch. Before editing, Codex should run a quick current-state verification against the local checkout.

---

## 1. Executive synthesis

The project has converged to a coherent thesis/system direction:

> **Entity-aware, quality-driven, multi-step next-best-view planning for egocentric Aria/ASE reconstruction, built first on oracle RRI labels and discrete candidate scoring, then extended toward stochastic rollouts, offline value learning, VLM-guided global subgoals, and simulator-backed online training.**

The core thesis should not be framed as “we already solve full continuous RL NBV.” The defensible center is:

1. **Oracle RRI supervision for ASE.** Render candidate depth from ground-truth meshes, backproject to point clouds, fuse with current semi-dense SLAM points, and score reconstruction improvement with Chamfer-style point↔mesh error.
2. **VIN-style learned candidate scoring.** Train a lightweight RRI scorer on frozen EVL/EFM3D features, candidate pose encodings, semi-dense projection statistics, voxel evidence, and a CORAL ordinal head.
3. **Non-myopic planning as the next layer.** Move from one-step greedy candidate ranking to bounded multi-step counterfactual rollouts using temperature-softmax sampling, Gumbel-Top-k diversity, and later offline Q/IQL/CQL value learning.
4. **Entity-aware extension.** Restrict RRI to target object/entity regions using GT or predicted OBBs, enabling task-conditioned NBV: “which next view improves this object?”
5. **Global semantic/VLM layer.** Use a VLM or LLM action model for long-horizon semantic subgoals over grounded map elements such as portals, rooms, frontiers, and entities; do not let it output raw continuous camera poses.
6. **Simulator layer.** If online training becomes necessary, prefer an ASE-native simulator if accessible; otherwise Isaac Sim is the strongest public full-modality candidate, Habitat-Sim is the fast geometry/RL sidecar, iGibson is useful for auxiliary modalities, and AI2-THOR/ProcTHOR is mainly for semantic/global planning.
7. **Repo hygiene is a blocker.** Before adding big features, fix scaffold slop, public/internal docs mixing, copied wrong-repo skills, tracked artifacts, split-brain `data` vs `data_handling`, duplicate cache/type contracts, and plotting/helper sprawl.

---

## 2. Canonical thesis formulation

### 2.1 Current paper/system state

The current ARIA-NBV paper positions the system as **quality-driven next-best-view planning with egocentric foundation models**. It extends the VIN-NBV idea of predicting **Relative Reconstruction Improvement (RRI)** to Aria Synthetic Environments (ASE), using:

- ASE snippets with synchronized Aria RGB/SLAM streams, trajectories, semi-dense points, and ground-truth meshes for a 100-scene subset.
- Frozen EFM3D/EVL features as egocentric voxel evidence.
- Candidate view generation around a reference rig pose.
- GT-mesh rendering for candidate depth.
- Backprojection and point fusion to compute oracle RRI.
- VINv3 candidate scorer with pose encoding, voxel evidence, semi-dense projection statistics, optional projection-grid CNN, voxel-projection FiLM, and CORAL ordinal prediction.

The current implementation is best described as a **one-step candidate-ranking system**, not yet an end-to-end NBV policy.

### 2.2 Central problem definition

At time step `t`, the system has:

- current reconstruction point set `P_t` from semi-dense SLAM points,
- GT mesh `M_GT` during oracle-label generation,
- a candidate pose set `Q_t = {q_1, ..., q_N}`,
- optionally a target entity `e`,
- history of selected views / observations.

The current one-step oracle objective is:

```text
q* = argmax_q RRI(q)
```

where:

```text
RRI(q) = [CD(P_t, M_GT) - CD(P_t ∪ P_q, M_GT)] / [CD(P_t, M_GT) + ε]
```

The desired next-stage objective is:

```text
τ* = argmax_{q_t, ..., q_{t+L-1}} cumulative quality improvement
```

with bounded counterfactual planning, not exponential tree search.

### 2.3 Proposed thesis contribution set

A strong and defensible contribution set:

1. **Oracle RRI pipeline for egocentric ASE.** Candidate generation, GT-mesh rendering, depth backprojection, point↔mesh Chamfer scoring, cache generation.
2. **VINv3 learned RRI scorer.** Frozen EVL/EFM backbone, candidate-conditioned projection evidence, ordinal RRI training, ranking/evaluation diagnostics.
3. **Stochastic multi-step counterfactual rollout module.** Temperature-softmax candidate selection, Gumbel-Top-k beam diversity, deterministic replay, full trace logging, `O(B·L·N)` complexity.
4. **Entity-aware RRI extension.** OBB-cropped object/entity RRI, entity-conditioned candidate scoring, object-centric candidate generation.
5. **Diagnostics and reproducibility.** Cache manifests, W&B/Optuna report generation, Streamlit/Rerun visual diagnostics, CI/pre-commit, issue roadmap.
6. **Future layer: VLM/global semantic planner and simulator-backed online learning.** Kept as scoped extensions unless implemented and validated.

---

## 3. Core math and definitions

### 3.1 Chamfer-style point↔mesh reconstruction error

For point set `P` and GT mesh surface `M_GT` with triangle faces `F_GT`:

```text
CD(P, M_GT) = A(P, M_GT) + C(P, M_GT)
```

Accuracy term:

```text
A(P, M_GT) = (1 / |P|) Σ_{p∈P} min_{f∈F_GT} d(p, f)^2
```

Completeness term:

```text
C(P, M_GT) = (1 / |F_GT|) Σ_{f∈F_GT} min_{p∈P} d(p, f)^2
```

Interpretation:

- **Accuracy** penalizes noisy or misregistered points that do not lie on the GT surface.
- **Completeness** penalizes missing surface regions and holes.
- NBV usually cares strongly about completeness, but both are needed to avoid degenerate noisy point additions.

### 3.2 Relative Reconstruction Improvement

For candidate `q`, render/backproject candidate point set `P_q`; fuse with current reconstruction:

```text
P_{t∪q} = P_t ∪ P_q
```

Then:

```text
RRI(q) = [CD(P_t, M_GT) - CD(P_t ∪ P_q, M_GT)] / [CD(P_t, M_GT) + ε]
```

Positive RRI means candidate acquisition improves reconstruction quality.

### 3.3 Entity-aware RRI

For target entity `e` with oriented bounding box `B_e`, define a margin-expanded box:

```text
B_e^{+δ}
```

Entity mask:

```text
χ_e(x) = 1[x ∈ B_e^{+δ}]
```

Cropped current points, candidate points, and mesh:

```text
P_t^e = {p ∈ P_t | χ_e(p)=1}
P_q^e = {p ∈ P_q | χ_e(p)=1}
M_GT^e = M_GT ∩ B_e^{+δ}
```

Entity RRI:

```text
RRI_e(q) = [CD(P_t^e, M_GT^e) - CD(P_t^e ∪ P_q^e, M_GT^e)] / [CD(P_t^e, M_GT^e) + ε]
```

This is the cleanest mathematical definition for the entity-aware thesis extension. It enables object-specific planning: improve the chair/table/door/cabinet of interest rather than the whole scene.

### 3.4 CORAL ordinal RRI training

RRI is continuous but skewed and stage-dependent, so direct regression is brittle. Discretize into `K` ordered bins.

Quantile edges:

```text
e_k = Quantile({r_i}, k/K), k ∈ {1, ..., K-1}
```

Ordinal label:

```text
y(r) = Σ_{k=1}^{K-1} 1[r > e_k]
```

CORAL targets:

```text
t_k = 1[y > k], k ∈ {0, ..., K-2}
```

CORAL cumulative probabilities:

```text
p_k = P(y > k)
```

Class marginals from cumulative probabilities:

```text
π_k = p_{k-1} - p_k, with p_{-1}=1 and p_{K-1}=0
```

Expected metric prediction using representative bin values `u_k`:

```text
r_hat = Σ_k π_k · u_k
```

Do **not** treat CORAL cumulative probabilities as class posteriors. They must be converted to class marginals before expected-RRI decoding.

### 3.5 Softmax stochastic rollout policy

For candidate scores `S_i = S(s_t, q_i, e)`, define:

```text
π_τ(q_i | s_t, e) = exp(S_i / τ) / Σ_j exp(S_j / τ)
```

Temperature behavior:

- `τ → 0`: greedy argmax.
- moderate `τ`: local exploration around high scores.
- high `τ`: near-random diversity.

### 3.6 Gumbel-Top-k diversity

For sampling `B` diverse candidates without replacement:

```text
g_i = -log(-log u_i),  u_i ~ Uniform(0,1)
score_tilde_i = log π(q_i | s,e) + g_i
```

Select top `B` by `score_tilde_i`.

Use Gumbel-Top-k especially for initial beam roots or per-step diversity.

### 3.7 Soft Q-learning bridge

Once trajectories are cached, learn multi-step value:

```text
Q(s, q, e)
```

Soft value over current discrete candidate set:

```text
V(s,e) = α log Σ_{q∈Q(s)} exp(Q(s,q,e)/α)
```

Soft Bellman target:

```text
y_t = r_t + γ V(s_{t+1}, e)
```

Induced policy:

```text
π(q | s,e) = exp(Q(s,q,e)/α) / Σ_{q'} exp(Q(s,q',e)/α)
```

This makes the current temperature-softmax rollout sampler a direct precursor to a learned maximum-entropy Q policy.


---

## 3A. Current ARIA-NBV implementation contract from paper/transcripts

This section preserves low-level details that Codex should keep stable while planning edits.

### 3A.1 Dataset/cache state

Current documented data setup:

```text
ASE total scenes:                 100k synthetic indoor scenes
GT mesh subset used for oracle:   100 scenes
ATEK/EFM mesh-scene snippets:     4,608 snippet windows
Current offline oracle cache:     883 snippets from 80 GT-mesh scenes
Train/val split:                  706 / 177
Snippet length:                   20 frames at 10 Hz = 2 s
Snippet stride:                   10 frames = 1 s
RGB preprocessing:                240 × 240
SLAM grayscale preprocessing:     roughly 320 × 240
Semi-dense point budget:          padded/clipped to 50k per snippet frame; optional collapse/subsample
```

Codex should verify these numbers in the current paper/configs before relying on them, but they are the current transcript-level project assumptions.

### 3A.2 Coordinate and camera conventions

Hard rules:

```text
world frame: gravity-aligned
rig frame: headset/body frame
camera frame: Aria LUF = +x left, +y up, +z forward
image coordinates: origin top-left, u right, v down
pose notation: T_A_B maps frame B into frame A
candidate poses are world ← camera transforms unless explicitly inverted
PyTorch3D rendering expects world → view convention; conversions must be explicit
CW90 / rotate_yaw_cw90 corrections are display/model-contract sensitive and must not leak silently
```

Backprojection must remain consistent with PyTorch3D’s rasterizer convention. The existing paper describes converting pixel centers into PyTorch3D NDC-like coordinates using `s = min(H, W)`:

```text
x_ndc = -((u + 1/2 - W'/2) · 2/s)
y_ndc = -((v + 1/2 - H'/2) · 2/s)
p_world = Π^{-1}(x_ndc, y_ndc, depth, C_q)
```

This convention exists because backprojecting directly from pixel screen coordinates can disagree with the rasterizer for non-square images.

### 3A.3 Candidate generation contract

Current / representative candidate generation parameters from the paper/transcripts:

```text
N_q / final candidates:           60
r_min:                            0.5 m
r_max:                            1.8 m
elevation min/max:                -20° / 25°
yaw span:                         170°
yaw jitter δ:                     60°
pitch jitter δ:                   30°
roll jitter δ:                    0°
align_to_gravity:                 true
min_distance_to_mesh:             0.2 m
ensure_collision_free:            true
collision backend:                PyTorch3D
ray_subsample:                    32
step_clearance:                   0.1 m
ensure_free_space:                true
```

Sampling pattern:

```text
1. sample candidate center direction on S^2 or with forward-biased Power Spherical distribution
2. scale into yaw/elevation caps
3. sample radius uniformly between r_min and r_max
4. construct world candidate center around reference rig pose
5. build radial look-away or configured orientation
6. apply bounded yaw/pitch/roll jitter
7. prune by mesh clearance, straight-line path collision, and free-space bounds
```

### 3A.4 Candidate rendering contract

Representative depth-rendering config:

```text
resolution_scale:                 0.5
znear:                            0.001 m
zfar:                             20 m
cull_backfaces:                   true
faces_per_pixel:                  1
blur_radius:                      0
bin_size:                         0
dtype:                            float32
```

Depth hits are valid when:

```text
pix_to_face >= 0
znear < depth < zfar
finite depth
```

### 3A.5 Current VINv3 architecture contract

Current VINv3 stable components:

```text
frozen EVL voxel evidence
pose encoder using rig-relative candidate pose
6D rotation representation + learnable Fourier features
pose-conditioned global context via candidate queries attending pooled voxel tokens
semi-dense projection scalar statistics
optional semi-dense projection grid CNN
voxel-projection FiLM
MLP scoring head
CORAL ordinal head
```

Representative current configuration from the paper/transcripts:

```text
EVL voxel grid:                   48^3, roughly 4 m local cube
field dim:                        24
global pool grid:                 5
semi-dense projection grid:       12
max semi-dense points:            16,384
head hidden dim:                  192
head layers:                      2
ordinal bins:                     15
trajectory encoder:               disabled in current run
semi-dense projection CNN:        enabled
voxel-projection FiLM:            enabled
trainable params in VIN head:     74,104
EVL backbone:                     frozen
```

Scene field input channels mentioned in the paper/transcripts:

```text
V_occ_pr / occupancy probability
V_insurf / observed surface evidence
V_incount / observation counts
V_normcount = log(1 + V_incount) / log(1 + max(V_incount))
V_cent_pr / centerness probability
optional V_infree, V_unknown, V_new
V_unknown = 1 - V_normcount
V_new = V_unknown ⊙ V_occ_pr
```

Semi-dense projection stats:

```text
coverage ratio
empty fraction
visibility fraction
valid projection fraction
valid depth mean
valid depth standard deviation
optional reliability weighting from observation count and inverse-distance uncertainty
```

Representative current training dynamics reported:

```text
best run id mentioned:            rtjvfyyp / v03-best
validation relative CORAL loss:   0.743 → 0.666
validation Spearman:              0.254 → 0.501
```

Codex should treat these numbers as paper/report facts to verify, not as automatically current benchmark truth.

---

## 4. Multi-step counterfactual rollout contract

### 4.1 Core objective

Implement **discrete multi-step counterfactual NBV rollout** on top of the current candidate-based pipeline.

At each step:

1. Take current counterfactual state.
2. Enumerate currently valid candidates.
3. Score all valid candidates with either oracle RRI, predicted RRI, or learned Q.
4. Convert scores to temperature-softmax probabilities.
5. Sample exactly one next candidate for that rollout.
6. Materialize the selected counterfactual state.
7. Append full diagnostics to trajectory trace.
8. Repeat up to horizon `L` or early termination.

### 4.2 Beam-width semantics

Beam width `B` means:

```text
B independently sampled rollout chains
```

It does **not** mean full branching with `B` children at every node.

Complexity must remain:

```text
O(B · L · N)
```

not:

```text
O(N^L)
```

If the first step is shared before splitting, score evaluations can be:

```text
N + B · (L - 1) · N
```

If all rollouts start independently:

```text
B · L · N
```

### 4.3 Scoring backend interface

Planner must not care whether scores come from:

- `oracle`: oracle RRI / entity RRI.
- `model`: predicted RRI from VIN scorer.
- `q_model`: learned multi-step Q value.
- `hybrid`: predicted RRI plus validity, uncertainty/Fisher, semantic/entity bonus, path cost.

Recommended interface:

```python
class CandidateScorer(Protocol):
    def score(self, state, candidates, target_entity=None) -> CandidateScoreTable:
        ...
```

### 4.4 Performance rule

Do not generate full counterfactual modalities for every candidate unless strictly needed by the scorer.

Correct separation:

```text
score all N candidates cheaply
materialize expensive render/fusion/modalities only for selected candidate per rollout step
```

Wrong implementation:

```text
render/fuse/generate full modalities for all N candidates at every rollout step
```

### 4.5 Required rollout parameters

Expose:

```text
beam_width
horizon
temperature
num_candidates
sampling_seed
score_backend = {oracle, model, q_model, hybrid}
sample_without_replacement
greedy_at_eval
top_k / top_p truncation optional
early_stop_score_threshold optional
```

### 4.6 Required per-step outputs

For every rollout step, store:

```text
rollout_id
step_index
state_id / snippet_id
candidate_ids_considered
valid_mask
invalid_reason per candidate
candidate_scores
candidate_softmax_probabilities
score_entropy
selected_candidate_id
selected_candidate_pose
selected_candidate_probability
selected_candidate_raw_score
temperature
counterfactual modalities for selected step
per-step oracle/predicted RRI
accuracy/completeness components
cumulative RRI / cumulative Chamfer summary
state update summary
random seed / RNG state sufficient for replay
```

### 4.7 Required per-rollout outputs

For each of `B` trajectories:

```text
selected pose sequence
full candidate tables per step
counterfactual state trace
all generated selected-step modalities
per-step and cumulative metrics
rollout diversity metrics
termination reason
```

### 4.8 Termination criteria

Support:

- fixed horizon `L`,
- no valid candidates,
- score improvement below threshold,
- collision / infeasible transition,
- acquisition budget,
- time/path-length budget,
- entity completeness threshold.

### 4.9 Metrics to log

At minimum:

```text
per-step RRI
cumulative RRI
accuracy term
completeness term
final Chamfer-style distance
rollout diversity across beams
score entropy per step
oracle-vs-predicted rank agreement
validity/collision rates
path length / travel cost
candidate-set score histograms
```

---

## 5. Entity-aware NBV design

### 5.1 Why entity-aware RRI matters

Scene-level RRI can prefer walls, floors, or irrelevant clutter. Entity-aware RRI asks:

> Which candidate improves the target object/entity I care about?

This is critical for AR guidance, object inspection, and task-aware reconstruction.

### 5.2 Invalidity is not the lowest ordinal bin

Do **not** encode invalid candidates as just the lowest CORAL class.

Invalidity is a constraint, not an ordered quality value. Different invalid causes are semantically different:

```text
collision
outside bounds
no target visibility
bad frustum
unsafe transition
no depth hits
out of EVL voxel extent
```

Recommended heads:

```text
rri_head:       RRI / ordinal RRI prediction
validity_head:  p(valid)
reason_head:    invalid reason distribution
```

Planner score:

```text
S(q) = RRI_hat_e(q)
       - λ_invalid · (1 - p_valid(q))
       - λ_collision · p_collision(q)
       - λ_motion · motion_cost(q)
```

### 5.3 Target entity selection

Object eligibility score:

```text
E(e) = w1·visible(e)
     + w2·partial_reconstruction(e)
     + w3·projected_area(e)
     - w4·distance(e)
     + w5·complexity(e)
```

Practical filters:

```text
min projected area > threshold
distance < threshold
visible surface fraction in [v_min, v_max]
not already fully reconstructed
enough GT mesh faces after OBB crop
OBB not too loose / not mostly background
```

### 5.4 Object-centric candidate generation

Use a mixture sampler:

```text
p(q | s,e) = λ_lookat · p_lookat(q | e)
           + λ_default · p_default(q | s)
           + λ_frontier · p_frontier(q | s,e)
```

#### Object look-at sampler

Target point:

```text
z_e = OBB center
```

or missing-surface centroid:

```text
z_e = Σ_{x∈missing(e)} w_x x / Σ_x w_x
```

Sample camera center on shell:

```text
c_q = z_e + r · u
r ~ U(r_min, r_max)
u ~ S^2
```

Orient toward target:

```text
R_q = LookAt(c_q, z_e) ∘ R_jitter
```

#### Default sampler

Keep the existing current-pose shell sampler to preserve scene exploration and avoid object-only myopia.

#### Frontier / missing-surface sampler

Bias candidates toward missing entity surfaces, normal directions, or frontier regions.

### 5.5 Entity-conditioned VIN

Entity descriptor:

```text
h_e = EncodeOBB(center, rotation, extent, class, confidence, current_completeness)
```

Candidate scorer:

```text
RRI_hat_e(q) = f_theta(F_EVL, h_e, h_q, projection_features, history)
```

Loss:

```text
L = L_CORAL
  + λ_reg · L_Huber
  + λ_valid · L_BCE
  + λ_reason · L_CE
```

### 5.6 Entity-aware diagnostics

Add visual/debug surfaces:

- selected OBB / GT and predicted boxes,
- OBB-cropped current points,
- OBB-cropped candidate points,
- cropped GT mesh,
- entity-level accuracy/completeness/RRI,
- scene-level vs entity-level metric comparison,
- candidate ranking colored by entity RRI,
- selected rollout trajectory around target object.

---

## 6. Hestia synthesis for ARIA-NBV

Hestia is best viewed as a **control and representation design pattern** rather than a replacement objective.

### 6.1 What Hestia contributes

Hestia’s transferable ideas:

1. **Directional observability.** A voxel is not merely observed/unobserved; each of its six faces can be observed from different directions.
2. **Hierarchical action factorization.** First predict a look-at target point, then predict camera position conditioned on that point.
3. **Close-greedy training.** Use a small discount (`γ≈0.1`) so the policy prioritizes immediate useful acquisition and avoids spurious long-horizon credit assignment.
4. **Feasibility projection.** Project predicted actions to nearest collision-free positions rather than relying only on penalties.
5. **Large-scale simulator diversity.** Train across many diverse scenes to improve generalization.

### 6.2 Hestia state and directional face visibility

Hestia state:

```text
s_t = {I_t, M_t, G_t, L_t}
```

where `I_t` is current image, `M_t` is camera/height metadata, `G_t` is voxel grid, and `L_t` is look-at point.

Cumulative face visibility update:

```text
F_t = f_t OR F_{t-1}
```

Viewing direction from voxel center `p_{v_i}` to collision-free camera position `a'_t`:

```text
d_{v_i} = (a'_t - p_{v_i}) / ||a'_t - p_{v_i}||
```

Face visibility for six normals `n_{i,j}`:

```text
f_t(v_i, j) = 1[d_{v_i} · n_{i,j} > 0]
```

For ARIA-NBV, this should inspire richer **directional observation histograms** for:

- semi-dense points,
- EVL voxels,
- target entity tokens,
- surface patches / normals.

### 6.3 Hestia reward and close-greedy behavior

Reward decomposition:

```text
r_t = r_coverage(s_t,a_t) + r_constraint(s_t,a_t)
```

Coverage reward uses newly observed voxel faces:

```text
new_visible_faces / total_faces
```

For ARIA-NBV, replace face coverage with quality-aligned reward:

```text
r_t = RRI_e(q_t)
      - invalid penalties
      - motion/path penalties
```

Use low discount initially:

```text
γ ∈ [0.1, 0.5]
```

### 6.4 Hestia-inspired ARIA architecture

Proposed VINv4 / policy hybrid:

1. **Target proposal head** predicts a look-at point or entity/missing-surface centroid.
2. **Target-conditioned local feature read** samples EVL voxel/scene features at the target point.
3. **Continuous relative-translation policy** proposes a feasible camera displacement.
4. **Feasibility projector** snaps/projsects to collision-free pose.
5. **RRI/entity-RRI reward** remains the quality objective.

Do **not** copy Hestia’s face-coverage objective as the final target. Use it as auxiliary/directional evidence only.

---

## 7. VLM / LLM global planning layer

### 7.1 Clean decomposition

Use VLM/VLA/LLM for **global semantic planning**, not low-level metric control.

Recommended hierarchy:

```text
VLM global planner:      choose grounded subgoal
Local NBV controller:    choose safe metric viewpoints
Verifier/replanner:      update memory and replan if needed
```

The VLM should output grounded symbolic/subgoal actions, not raw camera poses.

Example subgoals:

```json
{
  "subgoal_type": "cross_portal",
  "portal_id": "doorway_7",
  "target_region": "kitchen",
  "reason": "large unexplored connected free space behind portal"
}
```

```json
{
  "subgoal_type": "inspect_entity",
  "entity_id": "cabinet_12",
  "view_hint": "front-left-close",
  "reason": "target object or container likely relevant"
}
```

### 7.2 What the VLM should see

Do not feed raw point clouds alone. Provide a grounded multimodal scene memory:

- recent RGB views,
- portal/doorway descriptors,
- frontier descriptors,
- entity/object descriptors,
- region/room descriptors,
- local geometry summary,
- task/query.

### 7.3 Pointcloud/map descriptor token families

#### Portal tokens

```text
portal_id
center / normal / width / height
connected free-space components
clearance / traversability
explored ratio on far side
uncertainty behind portal
visibility evidence through portal
semantic hints from nearby images
estimated room-transition prior
```

#### Frontier tokens

```text
frontier_id
location
free-space volume behind it
distance / path cost
expected visibility gain
expected semantic novelty
reconstruction uncertainty / RRI deficit
```

#### Entity tokens

```text
entity_id
class / caption / open-vocabulary label
confidence
OBB / centroid / extent
visibility completeness
directional observation coverage
occlusion level
task relevance
support/container relations
```

#### Region / room tokens

```text
region_id
semantic type guess
explored fraction
connectivity
object priors
clutter level
reconstruction quality summary
travel cost from current pose
```

#### Local geometry summary tokens

```text
free-space shape
narrow passage flags
stairs/steps/slopes/hazards
visibility bottlenecks
surface-normal / directional-observation histograms
near-field occlusion structure
```

### 7.4 Why doorways are a high-signal VLM use case

Doorways are topological bottlenecks and semantic transitions. A local greedy NBV policy may keep refining the current room because doorway-crossing has low immediate RRI. A VLM/global planner can reason that:

- a doorway likely leads to a new room,
- a requested object may be behind it,
- semantic hints suggest room type,
- crossing the portal changes the global information state.

### 7.5 Guardrails

- VLM outputs must be constrained to a schema.
- Every selected subgoal must reference existing grounded IDs.
- Local planner verifies feasibility and executes.
- The VLM should be called sparsely, at subgoal boundaries, not at every frame.
- If grounding fails, fall back to frontier/entity heuristics.

---

## 8. Fisher information and 3DGS NBV ideas

### 8.1 Fisher information intuition

RRI is an **output-space** improvement criterion:

> If I acquire/fuse this candidate, how much does reconstruction error improve?

Fisher information is a **parameter-space** learning criterion:

> If I trained on this candidate, how much would it constrain uncertain model parameters?

For model parameters `θ` and candidate view `q` producing observation `y_q`:

```text
p(y_q | θ, q)
```

Score gradient:

```text
g_q = ∇_θ log p(y_q | θ, q)
```

Fisher matrix:

```text
F_q = E[g_q g_q^T]
```

Cheap view score:

```text
S_FI(q) = tr(F_q)
```

More principled but expensive information gain:

```text
log det(F_current + F_q) - log det(F_current)
```

### 8.2 How Fisher complements RRI

Potential hybrid score:

```text
S(q) = α · RRI_hat_e(q)
     + β · U(q)
     + γ · semantic/entity info(q)
     - penalties
```

Where `U(q)` could be:

- gradient norm of the RRI head,
- Fisher trace approximation,
- entropy over CORAL bins,
- ensemble variance,
- disagreement between geometry and semantic heads.

### 8.3 Semantic/dynamic 3DGS NBV takeaway

The semantic/dynamic Gaussian-splatting NBV method decomposes view value into:

```text
S(q) = λ_g · S_geom(q)
     + λ_s · S_sem(q)
     + λ_d · S_dyn(q)
```

Transferable to ARIA-NBV:

```text
S(q) = λ_rri · RRI_hat_e(q)
     + λ_sem · U_entity(q)
     + λ_info · U_model(q)
     + λ_valid · S_valid(q)
```

Do not copy full 3DGS training loop for the thesis core. Borrow the **score decomposition** and **information-seeking exploration bonus**.

### 8.4 Deformation network summary

In dynamic 3DGS, the scene remains a set of canonical 3D Gaussians. A deformation network predicts time-dependent changes:

```text
(Δμ_i(t), ΔR_i(t), Δs_i(t)) = D_φ(μ_i, t)
```

Deformed Gaussian:

```text
G_i(t) = (μ_i + Δμ_i, R_i ∘ ΔR_i, s_i + Δs_i, α_i, c_i, f_i)
```

The deformation network operates on Gaussians; it does not replace them. Its relevance to ARIA-NBV is future dynamic/object-state planning: score views that reduce uncertainty about motion, object state, or temporal changes.

---

## 9. RL and offline value-learning plan

### 9.1 Do not start with PPO/SAC

Strong recommendation from all sessions:

```text
Start with stochastic greedy rollouts + oracle/model scorer.
Then learn offline Q/IQL from cached transitions.
Only later try PPO/SAC/continuous online RL.
```

Reason: oracle RRI is expensive; on-policy PPO would require too many interactions unless a fast simulator/surrogate exists.

### 9.2 Transition cache

Store transitions:

```text
s_t
candidate_set_t
valid_mask_t
target_entity_e optional
a_t selected candidate
r_t = oracle RRI or entity RRI
s_{t+1}
done
invalid_reason
counterfactual modalities
all candidate scores/probabilities
```

### 9.3 First learned value baseline

Masked discrete Q-function over variable candidate sets:

```text
Q_θ(s, q, e)
```

Soft Bellman loss:

```text
L_Q = [Q_θ(s_t,q_t,e) - (r_t + γ α log Σ_{q'} exp(Q_target(s_{t+1},q',e)/α))]^2
```

### 9.4 IQL

Use IQL when training from offline rollouts and wanting stability under dataset constraints.

Key components:

- expectile value regression,
- Q regression to `r + γ V(s')`,
- advantage-weighted behavioral cloning for policy extraction.

IQL is attractive because it avoids aggressive evaluation of unsupported out-of-dataset actions.

### 9.5 CQL

Use CQL if learned Q starts hallucinating high values for unsupported or invalid candidates.

Simplified penalty:

```text
L_CQL = α [log Σ_a exp Q(s,a) - E_{a~D} Q(s,a)]
```

This pushes down actions not supported by the dataset.

### 9.6 Other RL/trajectory methods

- **Double DQN / Rainbow DQN:** useful first masked discrete Q baseline; beware variable candidate sets.
- **AWAC:** future path from offline oracle data to online fine-tuning.
- **PPO:** relevant for GenNBV/Hestia-style online continuous training; sample hungry.
- **SAC:** future continuous 5-DoF pose proposal with maximum entropy.
- **Trajectory Transformer:** future sequence model over rollout traces, using beam/stochastic beam planning.
- **Decision Transformer:** later if conditioning on target cumulative RRI.
- **Diffuser:** future continuous trajectory generation, likely too heavy for thesis core.

---

## 10. Simulator and online training plan

### 10.1 Required online modality contract

ARIA-NBV does not merely need RGB-D. It needs either native support or derivable equivalents for:

1. **Aria-like multi-camera streams**
   - RGB,
   - SLAM-left,
   - SLAM-right,
   - synchronized intrinsics/extrinsics,
   - rig/world poses.

2. **Aria optics fidelity**
   - Aria uses FisheyeRadTanThinPrism; generic pinhole is not enough for faithful EVL compatibility.

3. **Counterfactual geometric supervision**
   - depth/ray distance,
   - instance/semantic segmentation,
   - GT mesh,
   - GT OBBs/categories,
   - scene-language/floor-plan structure.

4. **SLAM-like structural products**
   - semi-dense points,
   - visibility/observation tables,
   - inverse-distance uncertainty,
   - observation counts.

5. **Derived planning fields**
   - EVL-style voxel evidence,
   - free-space/occupancy,
   - SDF/normals,
   - directional visibility,
   - collision checks,
   - projection statistics.

The hard part is not RGB/depth rendering. The hard part is Aria optics, multi-camera synchronization, MPS-like semi-dense products, and EVL-compatible voxel evidence.

### 10.2 Simulator ranking

#### 1. ASE-native simulator / Meta internal ASE generation stack

Best possible fit if accessible. It already matches Aria/ASE conventions, sensors, trajectories, scene language, GT supervision, and MPS-like products. Publicly, ASE appears available as dataset/tooling, not necessarily as an interactive simulator.

Action: investigate access through Project Aria / ASE research channels.

#### 2. NVIDIA Isaac Sim

Best public “full multimodal” option.

Pros:

- multiple cameras,
- RGB/depth/pointcloud/semantic/instance/2D/3D boxes,
- normals/motion vectors,
- IMU/lidar options,
- online generation,
- camera distortion/fisheye/radial-tangential-thin-prism capabilities closer to Aria than pinhole-only simulators.

Cons:

- heavy integration,
- must build Aria rig definition,
- must build ATEK/EFM export shim,
- must generate MPS-like semi-dense products separately.

#### 3. Habitat-Sim

Best fast geometry/RL sidecar.

Pros:

- high-throughput RGB/depth/semantic rendering,
- fisheye variants,
- navmesh/pathfinding,
- semantic scene graph,
- good for frontier/doorway/topology experiments.

Cons:

- fisheye model mismatch with Aria,
- no native MPS-like semi-dense products,
- less sensor fidelity than Isaac.

#### 4. iGibson

Good auxiliary-modality sandbox.

Pros:

- RGB,
- normals,
- segmentation,
- 3D point clouds,
- depth,
- optical flow,
- scene flow,
- LiDAR.

Cons:

- weaker ARIA/ATEK/EFM fit,
- custom rig/semi-dense export still required.

#### 5. AI2-THOR / ProcTHOR

Best reserved for semantic/global planning experiments.

Pros:

- object metadata,
- interaction affordances,
- depth/segmentation,
- multiple camera views,
- useful for VLM subgoal planning.

Cons:

- weak Aria camera fidelity,
- weaker for exact geometry/RRI stack,
- not the main local counterfactual renderer.

### 10.3 Recommended integration phases

**Phase S0: ASE-native counterfactual sensor server.**

Use existing ASE assets and GT meshes to query novel poses and derive:

- depth,
- candidate point clouds,
- masks/instances,
- OBB projections,
- normals/SDF/collision,
- directional visibility,
- optional splat-rendered RGB.

**Phase S1: Habitat fast RL sidecar.**

Prototype topology/global/doorway/frontier behavior and run fast rollouts.

**Phase S2: Isaac full multimodal simulator.**

Build Aria rig, sensors, distortion model, data-export bridge, and eventually online policy training.

**Phase S3: VLM semantic planning simulator.**

Use AI2-THOR/ProcTHOR or Habitat scene graph to train/evaluate subgoal selection.

---

## 11. External papers and implementation pointers

### 11.1 Core NBV / ARIA papers

| Source                        | Role in ARIA-NBV                                                                                                      | Pointer                                                           |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Aria-NBV paper / main.pdf** | Current project statement, oracle RRI pipeline, VINv3 architecture, limitations.                                      | local `main.pdf`; repo paper under `docs/typst/paper/`            |
| **VIN-NBV**                   | Closest conceptual baseline: RRI, candidate ranking, greedy sequential NBV, CORAL ordinal labels.                     | https://arxiv.org/abs/2505.06219                                  |
| **GenNBV**                    | Continuous 5-DoF RL baseline; coverage-reward contrast; multi-source state embedding; simulator training.             | https://arxiv.org/abs/2402.16174                                  |
| **Hestia**                    | Voxel-face directional observability, hierarchical look-at/position action, close-greedy RL, feasibility projection.  | https://arxiv.org/html/2508.01014v3                               |
| **SceneScript**               | Structured scene language, ASE dataset, walls/doors/windows/OBBs, extendable commands, scene memory for VLM planning. | https://projectaria.com/scenescript                               |
| **EFM3D / EVL**               | Frozen egocentric voxel lifting backbone and ASE/AEO integration.                                                     | https://github.com/facebookresearch/efm3d ; Project Aria EVL docs |

### 11.2 Information gain / Gaussian splatting NBV

| Source                                                   | Use                                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **FisherRF**                                             | Fisher-information view selection; parameter-space expected information gain.                  |
| **Semantic/dynamic 3DGS NBV**                            | Decompose view value into geometry, semantic, deformation/dynamic information.                 |
| **OUGS**                                                 | Object-aware uncertainty in 3DGS; motivates object-centric view scoring.                       |
| **Informative Object-centric NBV for Object-aware 3DGS** | Object-centric cluttered-scene NBV; target-object information gain.                            |
| **POp-GS**                                               | Optimal experimental design / P-optimality for Gaussian-splatting NBV.                         |
| **SA-ResGS**                                             | Sparse-view active 3DGS and uncertainty stabilization.                                         |
| **ActiveGS / ActiveSplat**                               | Online mapping and view planning with Gaussian splatting.                                      |
| **Egocentric splats**                                    | Aria-specific photorealistic splat reconstruction; candidate RGB/counterfactual visualization. |

### 11.3 RL / planning papers

| Source                                    | Use                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| **Soft Q-Learning / maximum-entropy RL**  | Principled basis for softmax over Q values.                              |
| **Gumbel-Top-k / stochastic beam search** | Beam diversity and sampling without replacement.                         |
| **IQL**                                   | Offline value learning from cached oracle rollouts.                      |
| **CQL**                                   | Conservative protection against Q overestimation.                        |
| **Double DQN / Rainbow DQN**              | First masked discrete Q baseline.                                        |
| **AWAC**                                  | Offline-to-online fine-tuning path.                                      |
| **PPO**                                   | Online continuous control baseline, used by GenNBV/Hestia style systems. |
| **SAC**                                   | Continuous maximum-entropy actor-critic future direction.                |
| **Trajectory Transformer**                | Offline RL as sequence modeling; trajectory beam planning.               |
| **Decision Transformer**                  | Return-conditioned candidate sequence generation.                        |
| **Diffuser**                              | Future diffusion-based continuous trajectory synthesis.                  |

### 11.4 Semantic/global planning papers

| Source            | Use                                                               |
| ----------------- | ----------------------------------------------------------------- |
| **ConceptGraphs** | Open-vocabulary 3D scene graph for entity/room/object memory.     |
| **VLMaps**        | Language-indexed 3D maps.                                         |
| **SayPlan**       | LLM planning grounded in 3D scene graphs and replanning.          |
| **EgoLifter**     | Open-world egocentric 3D object segmentation with 3DGS/SAM masks. |
| **LangSplat**     | Language-aware Gaussian scene representation.                     |
| **SAM 2**         | Image/video segmentation for future open-world entity extraction. |

### 11.5 Architecture references

| Source       | Use                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------- |
| **QCNet**    | Query-centric encoding: encode scene once, decode many query/candidate scores.                     |
| **LMFormer** | Structured-context prioritization and iterative refinement; analogy to portals/entities/frontiers. |
| **RayTran**  | Posed image/voxel bidirectional attention used in SceneScript.                                     |

### 11.6 Libraries / external implementations

| Library / repo                       | Use                                                                 | Notes                             |
| ------------------------------------ | ------------------------------------------------------------------- | --------------------------------- |
| `projectaria_tools`                  | Aria calibration, data loading, MPS/semi-dense products, ASE tools. | Core ecosystem.                   |
| `facebookresearch/efm3d`             | EVL inference/training, ASE/AEO integration, ATEK format.           | Core backbone.                    |
| `PyTorch3D`                          | Mesh rasterization, cameras, point↔mesh distances, Chamfer.         | Core oracle.                      |
| `Open3D`                             | Mesh/point processing, sampling, visualization, simplification.     | Useful utility.                   |
| `WebDataset`                         | Scalable dataset shards for training/caches.                        | ATEK/EFM compatible.              |
| `gsplat`                             | CUDA 3DGS rendering.                                                | Future RGB/counterfactual splats. |
| `Nerfstudio / Splatfacto`            | Easier splat/NeRF experiments.                                      | Prototype only.                   |
| `facebookresearch/egocentric_splats` | Aria photorealistic splat reconstruction.                           | Future counterfactual RGB.        |
| `TorchRL`                            | Native PyTorch RL buffers/collectors.                               | Good for custom env.              |
| `d3rlpy`                             | Offline RL baselines.                                               | Fast IQL/CQL prototyping.         |
| `Stable-Baselines3`                  | PPO/SAC baselines once Gym env exists.                              | GenNBV/Hestia style.              |
| `CleanRL`                            | Single-file RL implementations for adaptation.                      | Good educational baseline.        |
| `Rerun`                              | Interactive 3D/temporal diagnostics; rollout flight recorder.       | Optional dependency only.         |
| `W&B`                                | Experiment tracking, sweeps, artifact logs.                         | Keep for final reports.           |
| `Optuna`                             | Hyperparameter sweeps.                                              | Keep constrained.                 |
| `DUSt3R / MASt3R`                    | RGB-only geometry fallback.                                         | Future real-world support.        |
| `Depth Anything V2`                  | Monocular depth fallback.                                           | Future RGB-only counterfactuals.  |
| `Habitat-Sim`                        | Fast geometry/RL simulator.                                         | Sidecar.                          |
| `Isaac Sim / Isaac Lab`              | Full multimodal simulator / online RL.                              | Heavy but strongest public.       |
| `iGibson`                            | Extra modalities (normals/flow/segmentation).                       | Auxiliary sandbox.                |
| `AI2-THOR / ProcTHOR`                | Semantic object interaction and global VLM planning.                | Not core local RRI renderer.      |

---

## 12. Rerun diagnostics decision

Use Rerun as the **3D flight recorder** for ARIA-NBV, not as the general plotting backend.

### Use Rerun for

- candidate frusta,
- selected trajectories,
- semi-dense point clouds,
- candidate point clouds,
- GT/cropped meshes,
- entity OBBs,
- per-step rollout visualization,
- scalar timelines for RRI/entropy/validity,
- `.rrd` offline sharing.

### Do not use Rerun for

- final thesis ablation plots,
- static metric curves,
- all Streamlit plotting,
- core RRI math dependencies,
- hard required package import.

### Integration pattern

Add optional module:

```text
aria_nbv/diagnostics/rerun_logger.py
```

Optional CLI:

```bash
uv run nbv-rerun-rollout --cache-path ... --sample-id ... --out outputs/rerun/sample.rrd
```

Add optional dependency:

```toml
[project.optional-dependencies]
diagnostics = ["rerun-sdk"]
```

Coordinate caution: Aria uses LUF camera frames; Rerun examples often use RDF. Add explicit conversion and tests.

---

## 13. Repo scaffold and docs cleanup

### 13.1 Scaffold diagnosis

`prml-vslam` has a better operational scaffold; `ARIA-NBV` has the richer research system. Transfer scaffold discipline, not domain content.

Good PRML patterns to port:

- README as project front door,
- SETUP.md for environment/runbook,
- `make ci`, lint/test/docs targets,
- pre-commit,
- issue/PR lifecycle skills,
- compact root `AGENTS.md`,
- package README/REQUIREMENTS contracts,
- structured agent backlog.

ARIA already has good domain-specific nested guides and memory docs. Keep those, but prune noisy surfaces.

### 13.2 Public docs problem

Public docs currently mix too many roles:

```text
thesis narrative
scratchpad ideas
roadmap/todos
questions
literature dumps
implementation notes
external implementation indices
agent scaffold mirrors
generated context
Typst/Quarto material
```

Target rule:

```text
public docs explain the project;
.agents helps agents work on the project;
generated context is generated and isolated.
```

### 13.3 Target docs shape

```text
docs/
  index.qmd
  setup.qmd
  contents/
    theory/
      rri_theory.qmd
    implementation/
      oracle_rri_pipeline.qmd
      vin_scorer.qmd
      evaluation_protocol.qmd
    experiments/
    literature/
      scene_script.qmd
      efm3d.qmd
      vin_nbv.qmd
      gennbv.qmd
      hestia.qmd
  reference/
  typst/
    paper/
    slides/
```

Move or delete:

- `todos.qmd` → GH issues / `.agents/memory/state/TODOS.md`.
- `roadmap.qmd` → GH milestones/issues or short public roadmap only.
- `questions.qmd` → `.agents/memory/state/OPEN_QUESTIONS.md` unless curated.
- `ideas.qmd` → `.agents/history/ideas/` unless explicitly public.
- `resources/agent_scaffold/` → `.agents/references/` or hidden generated mirror.
- `_generated/context/` → `.agents/generated/context/` or ignored.
- stale external implementation indices → curated references only.

### 13.4 Root README / SETUP / AGENTS targets

Root README should contain:

```text
mission
current status
main claim and non-claims
quick start
reproducibility path
data/cache contract
docs map
smoke run
known limitations
```

SETUP.md should own:

```text
base environment
GPU/PyTorch/PyTorch3D
EFM3D/ATEK
Project Aria data tools
data download
cache setup
W&B
Streamlit
Rerun optional
paper/docs build
common failures
```

Root AGENTS.md should be thin:

```text
sources of truth
repo map
default workflow
commands
hard rules
where to go next
```

Detailed workflows belong in skills.

### 13.5 Skills to add / fix

Replace copied/wrong skills and add ARIA-specific ones:

```text
repo-context-router / aria-nbv-context
oracle-rri-contracts
nbv-geometry-contracts
vin-training-diagnostics
counterfactual-rollout-planner
entity-aware-rri
docs-curator / docs-paper-sync
simplification-and-backlog
code-review
dataset-cache-ops
create-pr
gh-issue-lifecycle
agents-db, if actually backed by TOML DB files
```

Every skill must be ARIA-specific. Remove references to PRML VSLAM, litkg-rs, wrong paths, or wrong commands.

### 13.6 Oh My Codex / OMX

Use OMX only after scaffold cleanup, and only as optional local orchestration.

Policy:

```text
Canonical repo state = AGENTS.md + .agents/skills + .agents/memory + GH issues.
OMX may be user-local / opt-in.
Do not commit personal .codex/.omx runtime config.
Commit templates only if useful.
```

Good OMX pilot tasks:

- docs cleanup classification,
- multi-agent PR review,
- test-gap audit,
- large refactor planning.

Do not use OMX to compensate for dirty docs or missing CI.

---

## 14. Package slop / decluttering targets

### 14.1 Highest-priority kill list

| Target                             | Action                                                                                      | Reason                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `.logs/` tracked artifacts         | Delete from git, add to `.gitignore`, replace with artifact manifest/download instructions. | Checkpoints/model weights are not source code.                        |
| public docs nav                    | Remove TODOs/project state/questions/agent scaffold/generated pages from public nav.        | Public thesis site is noisy.                                          |
| `docs/_quarto.yml`                 | Rewrite readable nav from scratch.                                                          | Current config/nav exposes too much and may contain stale repo links. |
| root README                        | Rewrite as project front door.                                                              | Current command dump is insufficient.                                 |
| `aria_nbv.data` vs `data_handling` | Delete, migrate, or reduce old package to thin shim.                                        | Split-brain implementation.                                           |
| `data_handling` legacy layer       | Complete cutover and delete/quarantine `_legacy_*`.                                         | Legacy wrappers obscure canonical APIs.                               |
| `pyproject.toml`                   | Reformat and split extras.                                                                  | Giant dependency blob.                                                |
| `vin/model_v3.py`                  | Split and remove experiment-history sludge.                                                 | Core model too large and mixed-purpose.                               |

### 14.2 Split-brain package surface

Previously identified duplicates between `aria_nbv.data` and `aria_nbv.data_handling`:

```text
efm_views.py
efm_dataset.py
efm_snippet_loader.py
vin_oracle_types.py
vin_oracle_datasets.py
offline_cache.py / oracle_cache.py
vin_snippet_cache.py / vin_cache.py
vin_snippet_provider.py / vin_provider.py
offline_cache_store.py
mesh_cache.py
```

Action:

1. Pick canonical package, likely `data_handling`.
2. Move all authoritative contracts there.
3. Convert old `data` to explicit deprecation shims or delete.
4. Update imports repo-wide.
5. Add tests to prevent duplicate type import identities.

### 14.3 Duplicate types/contracts to consolidate

- `VinOracleBatch` and collation helpers.
- Oracle cache metadata/entry/sample types.
- VIN snippet cache metadata/entry/build result.
- Mesh cache contracts (`MeshProcessSpec`, `ProcessedMesh`).
- `SceneCoverage` / `CacheCoverageReport` orphaned in old package.

### 14.4 Helper sprawl

Move shared helpers to explicit shared modules:

- `build_vin_snippet_view`,
- `vin_snippet_cache_config_hash`,
- `empty_vin_snippet`,
- repeated path/cache validators,
- repeated plotting primitives.

### 14.5 Plotting duplication

Consolidate low-level helpers:

```text
_to_numpy
_plot_slice_grid
_histogram_overlay
_plot_hist_counts_mpl
_pretty_label
```

Keep shared plotting primitives in one module, e.g.:

```text
aria_nbv/utils/plotting.py
```

Domain-specific plotting modules can remain, but should import shared primitives and avoid exporting private helpers.

### 14.6 Other cleanup targets

- Delete or justify `lightning/lit_module_old.py`.
- Replace repeated app cache dataclasses (`DataCache`, `CandidatesCache`, `DepthCache`, `PointCloudCache`, `RriCache`) with generic/base container if structurally identical.
- Extract duplicated `cache_dir` validators.
- Remove hard-coded old `/home/jandu/repos/NBV/...` paths.
- Split dependencies into extras: `core`, `train`, `viz`, `dev`, `notebooks`, `efm`, `diagnostics`, `all`.

---

## 15. GitHub roadmap and issue register to end of September

This merges the issue roadmaps from the transcripts. Dates are from the prior roadmap and should be adapted to the current semester/calendar if needed.

### 15.1 Milestones

| Milestone                                 |    Target window | Exit outcome                                                          |
| ----------------------------------------- | ---------------: | --------------------------------------------------------------------- |
| M0 — Governance & scope lock              |     by early May | Milestones/labels/templates/project board exist; thesis scope locked. |
| M1 — Repo hygiene & reproducibility       |              May | README/SETUP/CI/docs/skills trustworthy.                              |
| M2 — Oracle/data correctness & scale gate |             June | Geometry, device, cache, oracle bottlenecks controlled.               |
| M3 — VIN baseline & controlled ablations  |        June/July | VIN is evidence-backed baseline.                                      |
| M4 — Non-myopic planning baselines        |      July/August | One-step, close-greedy, stochastic/beam, and RL scaffolds comparable. |
| M5 — Entity-aware / scale / reports       | August/September | Entity-aware labels/scorer prototype and final experiment reports.    |
| M6 — Thesis freeze & release              |    end September | Paper, slides, configs, release package, demo frozen.                 |

### 15.2 Issues: governance and scaffold

1. **Create roadmap milestones, labels, issue templates, and project board.**
2. **Lock thesis scope and September success criteria.**
3. **Triage open draft PR backlog and convert residual work into issues.**
4. **Rewrite README, add SETUP.md and CONTRIBUTING.md.**
5. **Add root `make ci`, package checks, docs checks, and pre-commit parity.**
6. **Separate public docs from internal agent/generated surfaces.**
7. **Prune stale Quarto pages and align docs with Typst paper/current code.**
8. **Replace copied/sloppy skills with ARIA-specific custom skills.**
9. **Add scaffold validator and agent DB for issues/todos/resolved work.**
10. **Pilot OMX/Codex templates without committing personal runtime config.**
11. **Create one-scene data/cache/training smoke tutorial.**

### 15.3 Issues: oracle/data correctness

12. **Centralize device selection and make oracle pipeline MPS/CUDA/CPU-safe.**
13. **Add pose-frame/CW90 consistency guard for VIN, rendering, and cached batches.**
14. **Fix candidate/RRI shuffling and known VIN batch shuffle failure.**
15. **Add offline cache config-hash metadata, manifest summary, and filtering.**
16. **Add online/extended offline sample-generation mode to training/datamodule.**
17. **Make `OracleRriLabeler` safe with DataLoader `num_workers > 0`.**
18. **Profile oracle throughput and define scaling budget.**
19. **Protect fine-detail supervision: mesh/point-cloud downsampling policy and ablation.**
20. **Run candidate-generation realism/generalization experiments.**
21. **Document RRI/Chamfer formulas in Typst slides and theory docs.**

### 15.4 Issues: VIN baseline

22. **Add torchmetrics and richer Lightning validation metrics.**
23. **Fix predicted-RRI reporting semantics and calibration plots.**
24. **Run controlled VIN ablation matrix with fixed seeds/splits/schedule.**
25. **Evaluate learnable CORAL bin centers/shifts and imbalance handling.**
26. **Test stage-aware features or stage-aware binning.**
27. **Prototype candidate-relative positional encoding and query-centric fusion.**
28. **Add semi-dense reliability and candidate-visibility embeddings.**
29. **Prototype RGB/DINOv2 or EVL neck feature projection for candidate views.**
30. **Harden EVL checkpoint/config loading options.**

### 15.5 Issues: non-myopic planning

31. **Define MDP contract for counterfactual non-myopic planning.**
32. **Implement one-step greedy, close-greedy, and beam/stochastic rollout evaluator.**
33. **Harden multi-step oracle RRI pipeline and cumulative-RRI plots.**
34. **Harden discrete-shell Gymnasium/SB3 RL baseline.**
35. **Use VIN as fast surrogate reward or critic for planning.**
36. **Prototype continuous pose proposal with feasibility projection.**

### 15.6 Issues: entity-aware / reports / release

37. **Scale oracle cache coverage within the 100 GT-mesh ASE scenes.**
38. **Add multi-anchor and multi-candidate-set augmentation.**
39. **Implement entity-aware RRI with OBB-cropped mesh and points.**
40. **Add snippet-level object/entity selection policy.**
41. **Add entity-conditioned VIN scorer baseline.**
42. **Extend Streamlit/Rerun diagnostics for entity-aware and rollout analysis.**
43. **Decide and scope SceneScript / semantic-global planner integration.**
44. **Build W&B/Optuna report generator for final experiment tables.**
45. **Write sim-to-real, device, and human-in-the-loop feasibility memo.**
46. **Freeze thesis narrative and synchronize README, Quarto, Typst paper, and slides.**
47. **Create reproducible final release package and smoke matrix.**
48. **Prepare final defense slides and demo storyboard.**

### 15.7 Issues: package decluttering

49. **Delete tracked `.logs/` artifacts and replace with artifact manifest.**
50. **Resolve `aria_nbv.data` vs `aria_nbv.data_handling` split-brain package surface.**
51. **Consolidate duplicate `VinOracleBatch` and collation helpers.**
52. **Consolidate oracle cache contract types.**
53. **Consolidate VIN snippet cache contract types.**
54. **Consolidate mesh cache contract types.**
55. **Migrate or delete orphaned old-package coverage/report logic.**
56. **Centralize shared VIN snippet helpers.**
57. **Deduplicate plotting primitives and stop exporting private helpers.**
58. **Extract duplicated `cache_dir` validator logic.**
59. **Delete or canonicalize `lightning/lit_module_old.py`.**
60. **Unify repeated Streamlit/app cache dataclasses.**
61. **Split `vin/model_v3.py` into focused modules.**
62. **Reformat and split `pyproject.toml` dependencies into extras.**

---

## 16. Immediate Codex plan-mode priorities

### 16.1 First verification pass

Before editing, Codex should inspect current `main`:

```bash
git status --short
find . -maxdepth 3 -name AGENTS.md -print
ls -la
ls -la .agents .agents/skills .agents/memory/state || true
ls -la docs docs/contents || true
ls -la aria_nbv/aria_nbv || true
git ls-files .logs | head
```

Then verify whether the transcript-identified issues still exist.

### 16.2 First PR: scaffold sanity

Scope:

- fix wrong copied skills,
- add/clean README/SETUP skeleton,
- add pre-commit and `make ci` skeleton,
- add docs-curator skill,
- add issue/PR lifecycle templates.

Acceptance:

- no skill references wrong repo names,
- `make ci` runs at least lightweight checks,
- public docs nav is not made worse,
- no generated artifacts added.

### 16.3 Second PR: package split-brain cleanup plan

Do not immediately delete everything. First produce a map:

```text
old module -> new module -> imports -> action
```

Then fix one contract family at a time.

### 16.4 Third PR: oracle/geometry contracts

Lock:

- pose frame semantics,
- CW90 display-only vs model/render/cache conventions,
- PyTorch3D NDC backprojection,
- candidate pose vs rig pose naming,
- cache schema/version signatures.

### 16.5 Fourth PR: rollout module

Build the minimum `O(B·L·N)` stochastic rollout module using current scorer/oracle interfaces.

Do not add Q-learning until rollout traces are stable.

### 16.6 Fifth PR: entity-aware RRI prototype

Implement OBB-cropped metric on a tiny subset and visualize. Do not train entity VIN until labels are trusted.

---

## 17. Decision gates for Codex/planning

Before committing to larger feature work, answer these explicitly:

1. **Canonical data package:** Is `data_handling` the sole authoritative data/cache package?
2. **Current thesis core:** Is the final thesis core one-step VIN + rollout/entity extension, or is RL required for the claim?
3. **Entity source:** Use GT OBBs first, EVL-predicted OBBs second?
4. **Candidate generation:** Keep current shell sampler as default and add object-centric mixture sampler?
5. **Invalid candidates:** Hard mask only, separate validity head, or penalty/projection?
6. **Rollout scorer:** Oracle first, predicted RRI second, learned Q third?
7. **Simulator:** ASE-native first, Isaac first, or Habitat sidecar first?
8. **Rerun:** Optional diagnostics extra only?
9. **VLM planner:** Stretch/future or scoped prototype?
10. **Docs:** Which files are public thesis docs vs internal memory?

---

## 18. Risks and mitigations

| Risk                                                      | Mitigation                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Oracle RRI too expensive for online planning              | Cache labels, profile bottlenecks, score cheaply, materialize only selected candidates.          |
| Learned scorer overfits to current candidate distribution | Candidate profile versioning, augmentation, rank metrics, cross-scene splits.                    |
| CORAL score misinterpreted                                | Always convert cumulative probabilities to marginals before expected value.                      |
| Invalid candidates pollute ordinal labels                 | Separate validity/reason heads and planner masks.                                                |
| Entity RRI unstable due to tiny crops                     | Minimum mesh faces/points thresholds, OBB margin, report scene/entity metrics separately.        |
| Frame convention bugs                                     | Explicit PoseTW/CameraTW contracts, NDC tests, Rerun conversion tests.                           |
| Repo slop slows agents                                    | Clean skills/docs/CI before large feature branches.                                              |
| VLM hallucinated subgoals                                 | Grounded schema, entity/portal IDs, local verifier.                                              |
| Simulator mismatch with Aria                              | Treat modality contract as first-class; build export shims; keep ASE-native labels as reference. |
| Overclaiming RL                                           | State clearly: stochastic rollout/offline value learning are extensions unless fully validated.  |

---

## 19. Minimal implementation skeletons

### 19.1 Rollout dataclasses

```python
@dataclass(frozen=True)
class RolloutConfig:
    beam_width: int
    horizon: int
    temperature: float
    num_candidates: int
    seed: int
    score_backend: Literal["oracle", "model", "q_model", "hybrid"]
    sample_without_replacement: bool = True
    greedy_at_eval: bool = False
    top_k: int | None = None
    top_p: float | None = None

@dataclass
class CandidateScoreTable:
    candidate_ids: list[str]
    poses_world_cam: Tensor
    valid_mask: Tensor
    invalid_reasons: list[str | None]
    raw_scores: Tensor
    probabilities: Tensor
    entropy: float

@dataclass
class RolloutStepTrace:
    rollout_id: int
    step_index: int
    score_table: CandidateScoreTable
    selected_candidate_id: str
    selected_pose_world_cam: Tensor
    selected_probability: float
    selected_raw_score: float
    rri_oracle: float | None
    rri_predicted: float | None
    accuracy: float | None
    completeness: float | None
    cumulative_summary: dict[str, float]
    artifacts: dict[str, Any]

@dataclass
class RolloutTrace:
    config: RolloutConfig
    snippet_id: str
    entity_id: str | None
    steps: list[RolloutStepTrace]
    termination_reason: str
```

### 19.2 Candidate score object

```python
@dataclass
class CandidateScore:
    oracle_rri: float | None = None
    pred_rri: float | None = None
    q_value: float | None = None
    fisher_geom: float = 0.0
    fisher_sem: float = 0.0
    fisher_dyn: float = 0.0
    entropy: float = 0.0
    validity_prob: float = 1.0
    collision_prob: float = 0.0
    motion_cost: float = 0.0

    def total(self, w: ScoreWeights) -> float:
        return (
            w.rri * choose_available(self.q_value, self.pred_rri, self.oracle_rri)
            + w.fisher_geom * self.fisher_geom
            + w.fisher_sem * self.fisher_sem
            + w.entropy * self.entropy
            - w.invalid * (1.0 - self.validity_prob)
            - w.collision * self.collision_prob
            - w.motion * self.motion_cost
        )
```

### 19.3 Entity-aware label cache fields

```text
snippet_id
scene_id
entity_id
entity_source = {gt_obb, evl_pred_obb}
entity_class
entity_confidence
obb_center
obb_rotation
obb_extent
obb_margin
candidate_pose_world_cam
candidate_valid
invalid_reason
rri_scene
rri_entity
accuracy_scene
completeness_scene
accuracy_entity
completeness_entity
candidate_depth_optional
candidate_points_optional
mesh_crop_stats
point_crop_stats
config_hash
```

---

## 20. Recommended final narrative for advisor/thesis

The clean story:

1. **Problem:** Coverage-based NBV can fail in cluttered egocentric indoor scenes because coverage is not reconstruction quality.
2. **Baseline insight:** VIN-NBV shows RRI-based candidate scoring improves object-centric reconstruction quality.
3. **Gap:** VIN-NBV is object-centric and not grounded in egocentric Aria/ASE foundation-model priors; GenNBV/Hestia provide continuous RL but optimize coverage/face-coverage rather than RRI.
4. **Our system:** Build oracle RRI labels in ASE using GT meshes and semi-dense Aria reconstructions; train EVL-conditioned VIN scorer.
5. **Extension:** Add bounded multi-step counterfactual rollouts using the same scorer/oracle stack, preserving tractable `O(B·L·N)` complexity.
6. **Entity-aware direction:** Define object-specific RRI via OBB-cropped metrics, enabling task-aware inspection.
7. **Future:** VLM semantic planner selects portals/entities/frontiers; local NBV controller optimizes metric view quality; simulator enables online training.

Do not overclaim full continuous RL unless implemented and evaluated.

---

## 21. Codex action checklist

### Must do before feature expansion

- [ ] Verify current `main` against transcript issues.
- [ ] Fix stale/wrong skills.
- [ ] Add or repair CI/pre-commit/README/SETUP.
- [ ] Remove tracked artifacts and docs nav noise.
- [ ] Decide canonical data package.
- [ ] Create issues/milestones/project board.

### Core research implementation

- [ ] Lock oracle RRI geometry/frame contracts.
- [ ] Add cache manifests/config hashes.
- [ ] Add richer VIN evaluation CLI/metrics.
- [ ] Implement stochastic rollout traces.
- [ ] Compare one-step greedy vs random vs oracle rollout vs model rollout.
- [ ] Implement entity-aware RRI on tiny subset.
- [ ] Visualize with Streamlit + optional Rerun.

### Stretch / future

- [ ] Offline Q/IQL/CQL from rollout traces.
- [ ] Directional observation histograms.
- [ ] Hestia-style target proposal head.
- [ ] VLM subgoal planner over portals/entities/frontiers.
- [ ] Habitat/Isaac simulator bridge.
- [ ] 3DGS/Fisher uncertainty baseline.

---

## 22. One-sentence handoff to Codex

Build ARIA-NBV toward a **clean, reproducible, quality-driven egocentric NBV stack**: first fix repo/docs/package slop, then stabilize oracle RRI and VIN evaluation, then implement tractable multi-step counterfactual rollouts, then add entity-aware RRI and optional offline value learning, while treating VLM planning, continuous control, Fisher/3DGS uncertainty, and full simulator-backed online RL as scoped extensions rather than thesis-core claims until validated.


---

## Appendix A. Expanded source and implementation map

This appendix lists the external sources and implementation pointers that appeared across the sessions. Codex does not need to read everything before every task; use it as a routing table.

### A.1 ARIA / Project Aria / current project

- ARIA-NBV repo: `https://github.com/JanDuchscherer104/ARIA-NBV`
- ARIA-NBV GitHub Pages: `https://janduchscherer104.github.io/ARIA-NBV/`
- Aria-NBV paper: local `main.pdf`; repo `docs/typst/paper/`
- Ideas scratchpad: local `ideas.qmd`; public `https://janduchscherer104.github.io/ARIA-NBV/contents/ideas.html`
- Project Aria tools: `https://github.com/facebookresearch/projectaria_tools`
- ASE docs: `https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset`
- ASE data format: `https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format`
- Aria camera intrinsic models: `https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models`
- EFM3D / EVL repo: `https://github.com/facebookresearch/efm3d`
- EVL docs: `https://facebookresearch.github.io/projectaria_tools/docs/open_models/evl`

### A.2 NBV and active reconstruction

- VIN-NBV: `https://arxiv.org/abs/2505.06219`
- GenNBV: `https://arxiv.org/abs/2402.16174`
- Hestia: `https://arxiv.org/html/2508.01014v3`
- Hestia WACV PDF pointer from transcript: `https://openaccess.thecvf.com/content/WACV2026/papers/Lu_Hestia_Voxel-Face-Aware_Hierarchical_Next-Best-View_Acquisition_for_Efficient_3D_Reconstruction_WACV_2026_paper.pdf`
- Scan-RL: `https://arxiv.org/abs/2008.12664`
- SCONE: `https://arxiv.org/abs/2208.10449`
- PC-NBV implementation: `https://github.com/Smile2020/PC-NBV`
- MACARONS: `https://arxiv.org/abs/2303.03315`
- ActiveRMap / information gain baselines: see VIN-NBV and GenNBV references.

### A.3 Gaussian splatting / Fisher / uncertainty NBV

- FisherRF: `https://arxiv.org/abs/2311.17874`
- 4D Gaussian Splatting: `https://arxiv.org/html/2310.08528v3`
- Semantic/dynamic 3DGS NBV: `https://arxiv.org/abs/2512.22771`
- OUGS object-aware uncertainty in 3DGS: `https://arxiv.org/abs/2511.09397`
- Object-centric NBV for object-aware 3DGS: `https://arxiv.org/abs/2602.08266`
- POp-GS: `https://arxiv.org/abs/2503.07819`
- SA-ResGS: `https://arxiv.org/abs/2601.03024`
- ActiveSplat / ActiveGS pointer from transcript: `https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/jin2025ral.pdf`
- 3DGS implementation library: `https://docs.gsplat.studio/`
- Nerfstudio Splatfacto: `https://docs.nerf.studio/nerfology/methods/splat.html`
- Egocentric splats: `https://github.com/facebookresearch/egocentric_splats`

### A.4 RL / offline RL / sequence models

- Soft Q-Learning / energy-based policies: `https://arxiv.org/abs/1702.08165`
- Gumbel-Top-k / stochastic beam search: `https://arxiv.org/abs/1903.06059`
- Double DQN: `https://arxiv.org/abs/1509.06461`
- PPO: `https://arxiv.org/abs/1707.06347`
- SAC: `https://arxiv.org/abs/1812.05905`
- AWAC: `https://arxiv.org/abs/2006.09359`
- IQL: `https://openreview.net/forum?id=68n2s9ZJWF8`
- CQL: `https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html`
- Decision Transformer: `https://arxiv.org/abs/2106.01345`
- Trajectory Transformer: `https://arxiv.org/abs/2106.02039`
- Diffuser: `https://arxiv.org/abs/2205.09991`

### A.5 Semantic mapping / VLM planning / open-world scene memory

- SceneScript: `https://projectaria.com/scenescript`
- ConceptGraphs: `https://concept-graphs.github.io/`
- VLMaps: `https://arxiv.org/abs/2210.05714`
- SayPlan: `https://arxiv.org/abs/2307.06135`
- EgoLifter: `https://arxiv.org/abs/2403.18118`
- LangSplat: `https://arxiv.org/abs/2312.16084`
- SAM2: `https://github.com/facebookresearch/sam2`

### A.6 Architecture references

- QCNet: `https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf`
- LMFormer: `https://arxiv.org/abs/2504.10275`
- RayTran: referenced by SceneScript.
- CORAL ordinal regression: referenced in VIN-NBV and Aria-NBV paper; implementation pointer `coral-pytorch`.
- FiLM: feature-wise conditioning used by VINv3.

### A.7 Simulators

- Isaac Sim: `https://developer.nvidia.com/isaac/sim`
- Isaac Lab: `https://github.com/isaac-sim/IsaacLab`
- Habitat-Sim: `https://aihabitat.org/docs/habitat-sim/`
- Habitat-Lab: `https://aihabitat.org/docs/habitat-lab/`
- HM3D Semantics: `https://aihabitat.org/datasets/hm3d-semantics/`
- ReplicaCAD: `https://aihabitat.org/datasets/replica_cad/`
- iGibson: `https://svl.stanford.edu/igibson/`
- AI2-THOR: `https://github.com/allenai/ai2thor`
- ProcTHOR: `https://procthor.allenai.org/`
- ManiSkill: `https://maniskill.readthedocs.io/en/latest/user_guide/index.html`
- Genesis: `https://genesis-world.readthedocs.io/`

### A.8 Core engineering libraries

- PyTorch3D renderer: `https://pytorch3d.org/docs/renderer`
- Open3D: `https://www.open3d.org/docs/release/`
- WebDataset: `https://github.com/webdataset/webdataset`
- TorchRL: `https://docs.pytorch.org/rl/`
- d3rlpy: `https://d3rlpy.readthedocs.io/`
- Stable-Baselines3: `https://stable-baselines3.readthedocs.io/`
- CleanRL: `https://github.com/vwxyzjn/cleanrl`
- Rerun: `https://github.com/rerun-io/rerun`
- Rerun archetypes: `https://rerun.io/docs/reference/types/archetypes`
- Rerun operating modes: `https://rerun.io/docs/reference/sdk-operating-modes`
- W&B tracking: `https://docs.wandb.ai/models/track`
- DUSt3R: `https://arxiv.org/abs/2312.14132`
- Depth Anything V2: `https://depth-anything-v2.github.io/`

### A.9 Scaffold / agent tooling

- PRML-VSLAM repo to borrow scaffold patterns: `https://github.com/JanDuchscherer104/prml-vslam`
- Oh My Codex / OMX: `https://github.com/Yeachan-Heo/oh-my-codex`
- OpenAI Codex AGENTS.md guide: `https://developers.openai.com/codex/guides/agents-md`
- OpenAI Codex best practices: `https://developers.openai.com/codex/learn/best-practices`
- OpenAI Codex skills: `https://developers.openai.com/codex/skills`

---

## Appendix B. Transcript-to-section routing

Use this map if Codex needs to recover the source discussion:

```text
transcript-01 Turn 01/02: multi-step rollout requirement and acceptance criteria
transcript-01 Turn 03/04: RL paper survey and recommended RL method stack
transcript-01 Turn 05/06: semantic/dynamic 3DGS NBV paper breakdown
transcript-01 Turn 07/08: Fisher information and deformation network explanation
transcript-01 Turn 09/10: thesis notes, entity-aware RRI, papers/libraries, implementation plan
transcript-01 Turn 11/12: broad research, simulator search, online training context
transcript-01 Turn 13/14: GH issue roadmap
transcript-01 Turn 15/16: ruthless simplification and package/docs slop
transcript-01 Turn 17/18: Rerun plotting/diagnostics decision
transcript-02: scaffold cross-review, GH issues, ruthless simplification branch
transcript-03: scaffold cross-review, GH issues, ruthless simplification branch
transcript-04 Turn 01/02: unresolved main-branch issues list
transcript-04 Turn 03/04: Hestia literature review
transcript-04 Turn 05/06: VLM action/global planning layer
transcript-04 Turn 07/08: simulator modality requirements and ranking
```

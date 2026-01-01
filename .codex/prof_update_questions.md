# Questions for Professor Update Meeting (NBV + Oracle RRI + VIN)

```typst
= NBV + Oracle RRI + VIN — Open Questions for Update Meeting
// Sources: docs/contents/todos.qmd, docs/contents/questions.qmd, oracle_rri package exploration.

== 1) Dataset + Evaluation Setup (ASE constraints)
- Q: ASE provides a single recorded egocentric trajectory per scene. For NBV, is it acceptable to treat candidate poses as a *simulated* local action space (sample nearby SE(3) and render from GT mesh), or should we constrain candidates to what is plausibly reachable / consistent with the recorded trajectory?
- Q: Should we bake a motion cost into NBV (translation + yaw/pitch change), so the policy doesn’t always select “turn-around” views that are high-RRI but unrealistic?
- Q: What is the right evaluation story for the seminar: (a) RRI prediction quality (rank correlation / top-k recall) only, or (b) simulated NBV rollouts that show reconstruction improves over steps?

== 2) Oracle RRI Definition (Metric / Label Design)
- Q: Is point↔mesh Chamfer (accuracy+completeness) the right proxy for “reconstruction improvement” in this setting, or should we report/optimize alternatives (F-score@τ, visibility-weighted surface coverage, SDF-based distances, etc.)?
- Q: Should accuracy vs completeness be weighted differently? (E.g., completeness-heavy to reward discovering unseen surfaces vs accuracy-heavy for refining already-seen regions.)
- Q: Density mismatch: semi-dense SLAM points ``P_t`` are sparse, while candidate PCs from GT depth/mesh can be much denser. Do we need *explicit density equalization* (voxel downsample, stratified subsampling), or is the current setup acceptable for stable labels?
- Q: “Blank wall gets high RRI” can be a *metric/label design* issue (GT depth adds points, but real semi-dense SLAM may fail on low-texture). Should the oracle try to model that failure mode? If yes, how (texturedness weighting, uncertainty weighting, semidense-like sampling from depth, etc.)?
- Q: Mesh cropping / observability: is cropping the GT mesh to an AABB around ``P_t`` and ``P_q`` sufficient, or do we need a visibility/frustum-based crop to avoid penalizing other rooms / non-observable geometry?
- Q: Should the oracle incorporate occlusion/visibility explicitly (e.g., only count triangles visible from the current path / candidate frusta), or is the Chamfer setup “good enough” for the labeler?

== 3) Candidate View Generation (Sampling + Constraints)
- Q: Invalid candidates: should we *filter* collisions / too-close-to-mesh / no-line-of-sight views, or should we keep them and apply strong penalties so the model learns to avoid them?
- Q: Sampling support: do we allow full 360° azimuth around the reference (can yield high RRI but requires turning around), or restrict to a forward-biased hemisphere / limited yaw range to match realistic egocentric motion?
- Q: Roll: should we allow roll jitter / arbitrary roll angles, or keep roll fixed to reduce pose DOF and simplify the VIN input?
- Q: Gravity alignment: should we always sample in a gravity-aligned reference frame (yaw preserved, pitch/roll removed), or does that hide useful information for NBV (e.g., tilted camera situations)?
- Q: Discrete vs continuous actions: should we stay with VIN-style discrete candidate selection for now, or start shaping a path towards continuous pose regression (GenNBV-style) with free-space constraints?

== 4) VIN Inputs (What features actually matter?)
- Q: EVL features: which backbone outputs should VIN consume as “state”?
  - occ volume only vs occ+obb volumes
  - explicit free-space / occupancy probabilities vs learned neck features
  - OBB detections / semantics as extra conditioning for entity-aware NBV
- Q: Candidate conditioning: is sampling voxel features at the *candidate camera center* enough, or do we need a frustum-aware query (sample along rays / cone ring / multi-depth points) to predict RRI reliably?
- Q: Pose encoding: do we prefer shell descriptor + spherical harmonics (``u,f,r``) or learnable Fourier features on a 6D descriptor (``[t,f]``)? Any strong prior on what should generalize better across scenes?
- Q: Should we explicitly “project” / rotate features into the candidate view frame (or otherwise build equivariant conditioning), or is a strong pose embedding + voxel queries sufficient?

== 5) Training Objective (CORAL / ranking / calibration)
- Q: CORAL binning: how many bins should we use, and should thresholds be quantile-based? How sensitive is this to the candidate generation distribution (i.e., do thresholds need re-fitting when sampling changes)?
- Q: For NBV selection, should we train for *absolute* RRI prediction, or primarily for *ranking* (pairwise/listwise ranking losses, top-k objectives)?
- Q: Should we add an explicit “invalid candidate” class / head, or rely on masking + penalties?

== 6) Evaluation Metrics + Ablations (What to show in results)
- Q: What should be the primary headline metric: Spearman rank correlation of predicted score vs oracle RRI, top-1/top-k recall of the oracle-best candidate, or actual multi-step rollout gains in reconstruction metrics?
- Q: Which ablations are most compelling (and minimal): occ-only vs occ+obb, shell-SH vs LFF6D, center-sample vs frustum-query, forward-only candidates vs full 360°?
- Q: How should we handle GT mesh availability: report separately for the 100 “true GT mesh” val scenes vs pseudo-GT meshes for the rest?

== 7) Entity-Aware NBV (If we include it in the story)
- Q: Is per-entity reconstruction scoring meaningful in our setting? Should we compute entity RRI by masking mesh triangles inside OBBs and aggregating distances per entity?
- Q: How should entity weights be set (user-specified vs uncertainty-based vs task priors), and what is a convincing demo task for the seminar?

== 8) Practical / Implementation Questions (what to prioritize)
- Q: Depth semantics: are we fully confident PyTorch3D’s ``zbuf`` under ``in_ndc=False`` is “metric +Z depth” for our camera model, or should we validate/convert before relying on hit ratios and backprojections?
- Q: Visualization issues in diagnostics (e.g., missing axes in position sphere plot; pitch histogram not matching intuition): worth fixing now, or defer if the core pipeline is stable?
```


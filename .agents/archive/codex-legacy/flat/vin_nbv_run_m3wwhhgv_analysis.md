# VIN-NBV training run analysis (`m3wwhhgv`)

## What I inspected

- W&B run: `traenslenzor/aria-nbv/m3wwhhgv` (name `R2025-12-24_11-53-55`, created `2025-12-24T10:53:57Z`).
- Docs: `docs/contents/impl/vin_nbv.qmd`.
- Code:
  - `oracle_rri/oracle_rri/vin/model.py`
  - `oracle_rri/oracle_rri/vin/types.py`
  - `oracle_rri/oracle_rri/lightning/lit_module.py`

## Architecture recap (as implemented)

- Frozen EVL backbone → `EvlBackboneOutput` volumes.
- VIN “scene field” defaults: `["occ_pr", "occ_input", "counts_norm"]` → `Conv3d(1×1×1) + GroupNorm + GELU` → `field_dim=16`.
- Candidate features per pose:
  - Pose encoding: `ShellShPoseEncoder(u, f, r, s)` (default 64 dims).
  - Local field token: sample voxel field at `grid_size^2 * len(depths_m)` frustum points (default `4^2 * 4 = 64`) and masked-mean pool.
  - Optional global mean-pooled token (default enabled).
- Head: MLP → CORAL logits for `num_classes=15`.

## What the run history says (high-signal facts)

### 1) Loss scale suggests “near-random CORAL”

- `train/loss_step` (same as `train_loss_step`) over 0→1219:
  - mean ≈ **8.89**, median ≈ **8.71**, min ≈ **5.79**, max ≈ **15.74**.
- With CORAL (14 thresholds), a “uninformed” predictor often sits near `~0.69 * 14 ≈ 9.7` (all logits ≈ 0 → P(y>k)=0.5). This run spends most time in the **8–10** band → **little learning signal extracted**.

### 2) The logged “pred_rri_mean” is not on the same scale as `rri`

- `train/rri_mean_step` is true oracle mean RRI (continuous), and equals `(pm_dist_before - pm_dist_after) / pm_dist_before` exactly (checked).
- `train/pred_rri_mean_step` is **mean of `expected_normalized`** from CORAL (`E[y] / (K-1)`), i.e. a **rank/ordinal score in [0,1]**, not a metric RRI prediction.
- Practical consequence: comparing `rri_mean` vs `pred_rri_mean` in plots is misleading.

### 3) Model struggles most on “high improvement” batches

Across steps, `corr(train/loss_step, train/rri_mean_step) ≈ 0.74`.

Interpretation: whenever a snippet/candidate-set has larger oracle improvement potential, the model’s ordinal predictions are most inconsistent with the labels. This is consistent with a model that **under-predicts high bins** and/or lacks the features needed to recognize “new surface” opportunities.

### 4) Candidate validity reduces effective supervision

- `train/voxel_valid_fraction_step` mean ≈ **0.70** (min observed ≈ **0.09**, max **1.0**).
- So ~30% of candidates are dropped from the loss on average (mask uses `pred.candidate_valid & isfinite(rri)`), increasing gradient noise (especially when the candidate count is already low).

### 5) Data/label pipeline is brittle + expensive

W&B tables show **25** occurrences of:
- `Candidate generation produced 0 candidates...` (strict pruning / collision / free-space checks).

Also, `Pytorch3DDepthRenderer` logs an intrinsics “info” table ~**1225** times (nearly every step), creating **large logging overhead** on top of already expensive online oracle labeling.

## Likely root causes (ranked)

1. **Feature mismatch / underpowered scene signal**
   - The default scene field omits the most directly RRI-aligned feature: `new_surface_prior = unknown * occ_pr` (already supported by `_build_scene_field`, but not enabled by default).
   - With only `occ_pr`, `occ_input`, `counts_norm` and a 1×1×1 projection, VIN has to *learn* the multiplicative interaction that “unknown but occupied” implies.

2. **Metrics don’t diagnose ranking quality**
   - The current logs are mostly means. Without per-batch ranking metrics (Spearman, top‑k recall, “oracle-best hit rate”), the run can look “ok-ish” even when the model is not learning to rank candidates.

3. **High variance supervision**
   - Variable candidate count and ~30% validity masking mean each optimizer step sees a small, noisy set of supervised candidates.

4. **Candidate generator too strict**
   - Frequent “0 candidates” indicates the generator+rules reject many snippets, biasing the training distribution toward “easy” geometry / trajectories.

5. **Throughput constraints**
   - Online oracle labels + heavy W&B table logging make it hard to collect enough updates for a stable learning curve.

## Concrete next steps (actionable)

### Logging / diagnostics (highest leverage)

- Log label-space metrics:
  - `label_mean`, `label_hist` (ordinal labels after binning).
  - `pred_label_mean` (`expected`) + calibration plots.
- Log ranking metrics per batch:
  - Spearman rank corr between `pred.expected_normalized` and `rri` across candidates.
  - Top‑1 / top‑k recall: does the model’s best candidate fall in the oracle’s top‑k?
- Log “metric RRI prediction” derived from bins:
  - Map predicted expected class to an RRI proxy (e.g., bin midpoints from `rri_binner.json`) and log `pred_rri_proxy_mean` so it’s comparable to `rri_mean`.

### Model inputs

- Try `scene_field_channels=["new_surface_prior", "counts_norm", "free_input"]` (or add `unknown` explicitly) and re-run.
- Consider adding a second projection layer or a small 3×3×3 conv to let the model represent local spatial patterns before sampling.

### Data pipeline stability

- Loosen candidate pruning (reduce `min_distance_to_mesh`, disable `ensure_collision_free`/`ensure_free_space` temporarily) to reduce “0 candidate” cases and increase candidate variety.
- Reduce W&B “table per step” logging (only log on first step + every N steps, or log scalars).

## Repo hygiene note

- Fixed a Python parse issue in `oracle_rri/oracle_rri/vin/types.py` (unindented comments after `class EfmDict(TypedDict):`) so tooling like `make context` works again.


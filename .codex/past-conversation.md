
› Please provide element-wise shape informations and descriptions in oracle_rri/oracle_rri/vin/model_v3.py, following the example in (^1), *note* that the
  shapes denoted in (^1), might not be 100% correct!

  (^1):
  ```py
      def _encode_pose_features(
          self,
          pose_world_cam: PoseTW,
          pose_world_rig_ref: PoseTW,
      ) -> PoseFeatures:
          """Encode candidate poses in the reference rig frame.

          Args:
              pose_world_cam (PoseTW['B, Nq, 12']): SE(3) candidate camera poses in world frame ~ T_w^c.
              pose_world_rig_ref (PoseTW['B, 12']): SE(3) reference rig pose in world frame ~ T_w^r.
          Returns:
              PoseFeatures (dataclass):
                  pose_enc (Tensor['B, Nq, E']): Encoded pose features.
                  pose_vec (Tensor['B, Nq, D_v']): Pose vector features.
                  candidate_center_rig_m (Tensor['B, Nq, 3']): Candidate centers in rig frame.

          Relative pose encoding (R6D + LFF) avoids global-frame drift and was consistently stable in the vin-v2 sweep.
          """
  ```


• Added element‑wise shape + description blocks to the key VINv3 methods in oracle_rri/oracle_rri/vin/model_v3.py, following your example style. I used
  symbolic dims from docs/typst/shared/macros.typ (B, Nq, D/H/W, F_*), and kept hard constants only where stable (e.g., pose_vec = 9). I also replaced stale
  inline shape comments in _compute_global_context with symbolic ones.

  Updated docstrings include:

  - _ensure_vin_snippet
  - _prepare_inputs
  - _encode_pose_features
  - _build_field_bundle
  - _compute_global_context
  - _pool_voxel_points
  - _apply_film
  - _sample_semidense_points
  - _project_semidense_points
  - _encode_semidense_projection_features
  - _forward_impl

  Tests run:

  - uv run ruff format oracle_rri/oracle_rri/vin/model_v3.py
  - uv run ruff check oracle_rri/oracle_rri/vin/model_v3.py
  - uv run pytest oracle_rri/tests/vin/test_vin_model_v3_methods.py

  If you want Nq replaced by N everywhere (to exactly match the macro names), say the word and I’ll flip the notation.


› What are remaining potential causes for the bad performance and mode-collapse that we got in our first training run with oracle_rri/oracle_rri/vin/
  model_v3.py?


• Here are the most plausible remaining causes of the v3 mode‑collapse, ranked by likelihood and tied to what we’ve seen in the batched summary:

  1. Camera / pose frame mismatch (CW90 or T_world_cam alignment)

  - If candidate_poses_world_cam and p3d_cameras are not in the same frame or CW90 convention, projections go invalid → semidense_candidate_vis_frac ≈ 0 →
    candidate_valid mostly false.
  - Symptoms: valid mask mostly false in proj_data, coverage ~0, depth stats near zero/NaN.
  - Check: compare a few candidate_poses_world_cam to the p3d_cameras.R/T used; verify cw90_corrected and apply_cw90_correction are consistently applied.

  2. Projection space mismatch (NDC vs pixel space, image_size ordering)

  - transform_points_screen assumes pixel space when in_ndc=False. If image_size is swapped (W,H vs H,W) or in_ndc is wrong, most points are out‑of‑bounds.
  - Symptoms: valid mask near all‑false, coverage ~0, semidense_candidate_vis_frac ~0.
  - Check: log the fraction of valid points per candidate; verify p3d_cameras.image_size matches (H, W).

  3. Semidense points in wrong coordinate frame / scale

  - If VinSnippetView.points_world is not truly in world frame (or has wrong scale), projections fail silently.
  - Symptoms: projections mostly invalid, depth stats huge or degenerate.
  - Check: sample a few points vs candidate cameras; do a sanity plot of projections.

  4. RRI binning / label scaling mismatch

  - If rri_binner.json or label normalization differs from the cache, labels may collapse to a single bin → mode‑collapse.
  - Symptoms: training loss drops quickly, predictions collapse to one class; label histogram skewed.
  - Check: histogram of target bins for the first few batches.

  5. Candidate validity mask wipes most samples

  - candidate_valid requires pose_finite & voxel_valid_frac > 0 & semidense_candidate_vis_frac > 0. If any term is mostly false, training uses very few
    effective samples (or masks them entirely).
  - Symptoms: candidate_valid.mean() very small.
  - Check: log candidate_valid.mean(), voxel_valid_frac.mean(), semidense_candidate_vis_frac.mean().

  6. Voxel gating too aggressive

  - use_voxel_valid_frac_gate can zero out global_feat if coverage is low. With sparse voxels, this collapses the head input.
  - Symptoms: voxel_valid_frac near zero → global_feat near zero.
  - Check: gate outputs distribution; temporarily disable gating to isolate.

  7. Config mismatches vs best sweep trial

  - Even without PointNeXt, mismatched head_hidden_dim, head_num_layers, dropout, lr, weight_decay, semidense_proj_grid_size, semidense_proj_max_points, or
    obs_count/inv_dist_std stats can cause underfitting or saturation.
  - Check: diff the v3 run config vs the best trial (wsfpssd8) for all optimizer, head, semidense and voxel parameters.

  8. Backbone cache alignment issues

  - Cached t_world_voxel / voxel_extent must match the candidate poses and semidense points. If not, pos_grid and global attention are garbage.
  - Symptoms: global_feat uninformative; training collapses despite apparently valid projections.
  - Check: verify cached backbone outputs correspond to the same snippet + pose conventions.

  9. Semidense extras missing or mis‑ordered

  - If points_world lacks inv_dist_std / obs_count, weights become uniform → weaker signal. Not necessarily collapse, but can degrade.
  - Check: ensure points_world[..., 3] and [..., 4] are populated; obs_count should be >0.

  10. Too weak feature path (expected for “streamlined”)

  - v3 head currently uses only pose_enc + global_feat (no direct semidense_proj concat). If global features are weak, the head can collapse to a single
    bin.
  - This is the one “design” hypothesis rather than a bug. It shouldn’t fully collapse unless upstream signals are broken, but it can amplify collapse if
    they are.

  Quick triage (fastest to rule out):

  1. Log valid ratio, coverage, semidense_candidate_vis_frac, candidate_valid.mean() from the first 1–2 batches.
  2. Compare bin histogram of labels vs predicted bins for the same batches.
  3. Verify CW90/p3d alignment by projecting a few points manually for one candidate.

  If you want, I can add a small debug helper in v3 to print these stats per batch so you can pinpoint which failure mode is active.


› Here are the logs of train/coral_loss_rel_random_step: .logs/wandb_export_2026-01-26T14_35_46.608+01_00.csv please investigate and compare, note that
  some of the runs may have had better losses due to msking out samples with low visibility of the voxel features!


• Here’s what I found in .logs/wandb_export_2026-01-26T14_35_46.608+01_00.csv (46 runs total, 3336 steps max). I focused on train/coral_loss_rel_random_step
  and compared min, median, last, and run length:

  Key comparisons (loss)

  run                               steps  min     median  last
  vin-v3-01                           354  0.513  0.762    0.801   (rank ~26/46 by median)
  R2026-01-07_12-16-12_T41             267  0.587  0.739    0.757   (best median)
  R2026-01-07_12-16-12_T38             267  0.477  0.740    0.804
  R2026-01-07_19-09-57_T15             267  0.581  0.742    0.751
  NoValid+aux-loss-decay              3336  0.428  0.783    0.789   (very low min, higher median)
  no-valid+aux-loss-decay+traj-enc    2141  0.494  0.803    0.898
  traj-ecnoder+8classes               2020  0.510  0.822    0.768
  latest+fixes-2                      2543  0.566  0.819    0.767

  What this suggests (re masking note)

  - Best “min” values are from runs like NoValid+aux-loss-decay (min 0.428) but their medians are not better (0.783).
    These runs also show the largest min–median gaps, which is consistent with masking out hard/low‑visibility samples: you occasionally see very low
    losses, but the overall distribution doesn’t improve much.
  - The best median losses are around 0.739–0.742 (T41/T38/T15), while vin-v3-01 sits at ~0.762 (≈0.023 higher than best median).
    That’s mid‑pack, not catastrophic, but clearly not the best.
  - Several runs have very short length (9–89 steps); I would treat their “last” or “min” as unreliable for comparison.

  Implication for the mode‑collapse

  - The coral_loss_rel_random_step alone doesn’t show a dramatic failure for v3.
    So the collapse is likely driven by representation/validity issues rather than a loss explosion:
      - e.g., invalid projections, candidate_valid mask, or voxel gating removing most informative samples.

  If you want, I can pull a small list of runs with the largest min–median gap (most “mask‑suspect” behavior) and cross‑reference with their configs once
  you tell me where the config snapshots live in this export.

  Would you like a quick follow‑up that correlates coral loss with candidate_valid or semidense_candidate_vis_frac metrics (if those were logged)?


› Do some further analysis of the vin-v3-01 run against the best hisotric runs by considering more information that was logged to wandb - make use of the
  utils from oracle_rri/oracle_rri/configs/wandb_config.py


• I pulled the export at /.logs/wandb_export_2026-01-26T14_35_46.608+01_00.csv and used the metric‑pairing helpers from oracle_rri/oracle_rri/configs/
  wandb_config.py (loaded the _metric_pairs* block directly to avoid importing the full oracle_rri package). That utility confirms this export only contains
  one metric family:
  train/coral_loss_rel_random_step (plus _step). So any “further” comparison is limited to this loss; there are no val metrics or auxiliary stats in this
  CSV.

  vin‑v3‑01 vs best historic runs (train/coral_loss_rel_random_step)

  Top stable runs by median loss (>=200 steps):

  run                         steps  median   min     last   gap_med_min
  R2026-01-07_12-16-12_T41     267    0.7390   0.5869  0.7569 0.1520
  R2026-01-07_12-16-12_T38     267    0.7398   0.4767  0.8044 0.2631
  R2026-01-07_19-09-57_T15     267    0.7421   0.5814  0.7513 0.1607
  R2026-01-07_15-51-47_T5      267    0.7435   0.6085  0.7625 0.1350
  R2026-01-07_19-09-57_T11     267    0.7436   0.5675  0.7351 0.1761

  vin-v3-01:

  steps=354, median=0.7616, min=0.5132, last=0.8012
  first10%=0.7852, mid10%=0.7910, last10%=0.7522
  gap_med_min=0.2484

  Takeaways

  - vin-v3-01 is mid‑pack: median ≈0.7616 vs best median ≈0.7390 (+0.0226, ~3.1% relative).
  - It does not look catastrophically worse in this metric; the collapse likely shows in class distribution / confusion matrices, not directly in this loss.
  - Runs with very low min but higher median (large gap_med_min) are consistent with your suspicion about masking low‑visibility samples. The biggest gaps
    are:
      - NoValid+aux-loss-decay (gap 0.355)
      - no-valid+aux-loss-decay+traj-enc (gap 0.310)
      - traj-ecnoder+8classes (gap 0.312)
      - vin-v3-01 also has a moderate gap (0.248), so its low min does not reflect typical performance.

  What’s missing (and needed for deeper comparison)
  This export only has one metric. To compare “more information logged to W&B,” I need a CSV export that includes:

  - train/expected, train/candidate_valid_mean, train/semidense_candidate_vis_frac, or
  - val/coral_loss_rel_random_step, val/expected, etc.

  If you can export those metrics (or point me to a richer W&B export), I’ll extend the analysis with:

  - stability vs. visibility‑masking,
  - val/train gap,
  - correlation between candidate_valid and loss.

  If you want, I can also generate a small script to dump these extra metrics from W&B history using the helpers in wandb_config.py once you provide access
  or a richer export.


› The biggest issue is that the train-dynamics are really bad (no good improvement), and that the confusion matrices indicated full mode-collapse where it
  has been getting increasingly better for T41!


• Got it — the loss curve alone doesn’t tell the full story here, and the confusion matrix collapse is the stronger signal. Based on what we can see and how
  v3 is structured, here are the most plausible remaining causes for “bad dynamics + full collapse,” especially compared to T41:

  What the current export can and can’t tell us

  - The CSV you gave only contains train/coral_loss_rel_random_step.
    That metric is not sufficient to explain the collapse on its own, because it can look “OK” even when predictions collapse (e.g., due to masking or class
    imbalance).
  - So any deep comparison vs T41 needs more metrics (see below).

  Likely causes of the collapse (ranked)

  1. Effective sample mask too aggressive
      - v3 uses candidate_valid = pose_finite & voxel_valid_frac>0 & semidense_candidate_vis_frac>0.
        If this is mostly false, the loss sees very few valid candidates → the model collapses to a dominant bin.
      - T41 may have masked fewer samples (or had better visibility stats), improving both dynamics and confusion matrices.
  2. Global features are too weak / gated away
      - v3 only feeds pose_enc + global_feat into the head.
        If global_feat is gated to ~0 (low voxel_valid_frac or poor voxel coverage), the head reduces to a pose‑only predictor → easy collapse.
      - This is the most plausible architecture‑level failure given the new streamlined design.
  3. Projection pipeline still producing low visibility
      - Even if the code is correct, small shifts in intrinsics, image_size ordering, or candidate view distribution can make semidense projections sparse,
        which triggers the validity mask.
  4. Label distribution / binning mismatch
      - If bins are skewed or rri_binner.json doesn’t match the cache, CORAL can collapse while still showing “reasonable” loss values.
  5. Training config mismatch vs T41
      - T41 likely used a combination of settings that stabilized training (e.g., lower dropout, specific head sizes, visibility mask behavior, or different
        semidense_proj_* stats).
        v3‑01 is mid‑pack by median loss and not catastrophically bad, which suggests a few mismatched switches can cause the qualitative collapse.

  What we need to compare vin‑v3‑01 vs T41 properly
  Please export at least these W&B metrics (train + val if available):

  - train/candidate_valid_mean
  - train/semidense_candidate_vis_frac_mean
  - train/voxel_valid_frac_mean
  - train/expected or train/pred_bin_hist (if logged)
  - train/target_bin_hist (if logged)
  - val/coral_loss_rel_random_step
  - val/expected or bin histogram

  If you can export these, I’ll do a clean run‑by‑run comparison using the wandb_config utilities (pairing train/val step/epoch keys).

  Immediate, high‑value checks (to explain the collapse without new exports)

  1. Log candidate_valid.mean(), voxel_valid_frac.mean(), semidense_candidate_vis_frac.mean() for v3‑01 and T41.
  2. Log predicted bin histograms vs target bin histograms for the first few batches.
  3. Temporarily disable voxel gating (use_voxel_valid_frac_gate=False) and see if the collapse improves.

  If you want, I can add temporary logging for these metrics in the Lightning module so they show up in W&B (minimal patch).

  Which path do you want next?

  1. I add logging for candidate validity + visibility + bin histograms.
  2. You export the richer W&B CSV and I do a full comparison vs T41.


› Please compare all of these metrics! We are already logging all of them in oracle_rri/oracle_rri/lightning/lit_module.py, so they are already included in
  the wandb histories!


• Context compacted

• I pulled all scalar metrics logged by oracle_rri/oracle_rri/lightning/lit_module.py from W&B history (including train-aux/ and val-aux/). Confusion
  matrices and label histograms are logged as images (*-figures/...) so they don’t show up as numeric series. Below is a full metric-by-metric comparison of
  the latest logged values for vin‑v3‑01 vs the best run (T41, wsfpssd8). Note: vin‑v3‑01 has only a single logged point, so last == median for every
  metric.

  Train (scalars)

  metric                               vin-v3-01   T41        delta
  loss                                 7.657469    7.242287   +0.415182
  coral_loss                           7.463369    7.079302   +0.384068
  coral_loss_rel_random                0.769098    0.729520   +0.039578

  Val (scalars)

  metric                               vin-v3-01   T41        delta
  loss                                 7.599550    7.208847   +0.390703
  coral_loss                           7.375615    7.032352   +0.343262
  coral_loss_rel_random                0.760055    0.724681   +0.035373

  Train‑aux (all logged aux metrics)

  metric                               vin-v3-01   T41        delta
  aux_regression_loss                  0.019410    0.020122   -0.000712
  aux_regression_weight                10.000000   8.100000   +1.900000
  candidate_valid_frac                 0.782628    1.000000   -0.217372
  coral_loss_balanced_bce              0.942690    0.881852   +0.060838
  coral_loss_focal                     0.137784    0.123742   +0.014042
  coral_monotonicity_violation_rate    0.000000    0.000000    0.000000
  coverage_weight_mean                 0.857064    0.952355   -0.095291
  coverage_weight_strength             0.600000    0.200000   +0.400000
  pred_rri_mean                        0.101521    0.105538   -0.004017
  rri_mean                             0.108108    0.108108   ~0.000000
  semidense_candidate_vis_frac_mean    0.061877    0.082290   -0.020413
  semidense_candidate_vis_frac_std     0.047467    0.055480   -0.008012
  semidense_valid_frac_mean            0.061877    0.082290   -0.020413
  semidense_valid_frac_std             0.047467    0.055480   -0.008012
  spearman                             0.024762    0.328412   -0.303650
  top3_accuracy                        0.201020    0.285190   -0.084171
  voxel_valid_frac_mean                0.702216    0.702216   ~0.000000
  voxel_valid_frac_std                 0.383736    0.387583   -0.003847

  Val‑aux (all logged aux metrics)

  metric                               vin-v3-01   T41        delta
  aux_regression_loss                  0.022394    0.021789   +0.000604
  aux_regression_weight                10.000000   8.099999   +1.900001
  candidate_valid_frac                 0.785426    1.000000   -0.214574
  coral_loss_balanced_bce              0.937605    0.886429   +0.051176
  coral_loss_focal                     0.133245    0.121224   +0.012020
  coral_monotonicity_violation_rate    0.000000    0.000000    0.000000
  coverage_weight_strength             0.600000    0.200000   +0.400000
  pred_rri_bias2                       0.000037    0.000041   -0.000005
  pred_rri_mean                        0.100413    0.100036   +0.000377
  pred_rri_variance                    0.044750    0.043537   +0.001213
  rri_mean                             0.106476    0.106476    0.000000
  semidense_candidate_vis_frac_mean    0.065307    0.087345   -0.022038
  semidense_candidate_vis_frac_std     0.051472    0.060378   -0.008906
  semidense_valid_frac_mean            0.065307    0.087345   -0.022038
  semidense_valid_frac_std             0.051472    0.060378   -0.008906
  spearman                             0.185934    0.312216   -0.126282
  top3_accuracy                        0.190481    0.257449   -0.066968
  voxel_valid_frac_mean                0.696361    0.696361   ~0.000000
  voxel_valid_frac_std                 0.375792    0.380222   -0.004430

  Key takeaways from the full metric comparison:

  - Candidate validity + semidense visibility are much lower in vin‑v3‑01 (candidate_valid_frac ~0.78 vs 1.00; semidense visibility ~0.062 vs 0.082), which
    strongly correlates with the observed collapse.
  - Spearman/top‑3 accuracy are dramatically lower (train spearman 0.025 vs 0.328), consistent with no ranking signal.
  - Coverage weighting is effectively stronger in v3‑01 (coverage_weight_strength=0.6 vs 0.2), and aux regression weight is higher (10 vs 8.1), which
    suggests a config/HParam mismatch relative to the best sweep run.
  - Voxel validity is essentially identical, implying the voxel backbone itself is not the primary issue; the gap is driven by candidate/semidense
    projection validity and optimization weighting.

  If you want, I can produce the same “all-metrics” table for T38/T15/T11 and/or dump a CSV for the comparisons.


› What architectural differences might explain these deviations?


• Below are the concrete architecture differences between vin‑v3‑01 and the best sweep run T41 (wsfpssd8) that plausibly explain the metric gaps (low
  spearman/top‑3, lower candidate_valid_frac, lower semidense visibility, higher loss). I’ve tied each item to what the code actually does and to the W&B
  configs we just pulled.

  Key architectural deltas vs T41 (v2) that can drive the deviations

  - Semidense features are not fed into the head in v3.
    In oracle_rri/oracle_rri/vin/model_v3.py the head input is just pose_enc + global_feat; semidense_proj is computed but only used for candidate_valid
    (and for visibility logging). In v2/T41, semidense_proj (plus optional frustum context and point encoder features) is concatenated into the head. This
    removes the most candidate‑specific signal and makes collapse to mean much more likely.
  - Semidense frustum MHCA is removed entirely.
    T41 had enable_semidense_frustum=True, semidense_visibility_embed=True, and semidense_frustum_mask_invalid=True. That’s a candidate‑conditioned
    attention block over projected semidense points. In v3 it’s gone, so you lose a strong candidate‑specific inductive bias.
  - PointNeXt encoder path is removed.
    T41 had use_point_encoder=True with a 192‑dim PointNeXt S encoder and a FiLM modulation into the field. In v3 there is no semidense point encoder at
    all. This removes per‑scene geometry signals that helped T41 distinguish candidates.
  - Trajectory context is removed.
    T41 had use_traj_encoder=True and a trajectory attention block. v3 drops the entire trajectory context, removing temporal/motion cues that can be
    strongly correlated with RRI.
  - Field modulation uses a different signal.
    v2/T41 used semidense projection FiLM (sem_proj_film) to modulate the global field with candidate‑dependent semidense stats. v3 uses voxel‑projection
    FiLM (voxel_proj_film) computed from pooled voxel centers instead. This is a weaker proxy for candidate visibility and can reduce candidate
    differentiation.
  - Voxel validity handling flips from “feature” to “gate.”
    T41 uses use_voxel_valid_frac_feature=True and use_voxel_valid_frac_gate=False; v3 uses only a gate (use_voxel_valid_frac_gate=True) and doesn’t append
    the fraction as a feature. Gating can suppress gradients for partially observed candidates, especially when combined with weaker candidate‑specific
    signals, contributing to collapse.
  - Projection/statistics settings differ in a way that reduces visibility.
    T41 uses semidense_proj_grid_size=12, v3 uses 16. With a larger grid, coverage decreases (more bins → fewer occupied), which matches the observed drop
    in semidense_candidate_vis_frac_mean. Since v3 requires semidense_candidate_vis_frac > 0 for candidate_valid, this directly lowers candidate_valid_frac.
    Also, v3 hard‑codes log‑normalized obs_count weighting (and inv_dist_std weighting), while T41 used semidense_obs_count_norm=none. That shifts the
    visibility proxy and can deflate the fraction.
  - Head depth and field capacity differ.
    v3 run used head_num_layers=3 vs 1 in T41, and field_dim=16 vs 24. That’s a deeper MLP with less field capacity; with weaker inputs (no semidense/
    trajectory/point features) it can be harder to optimize and more prone to collapse.

  Why these line up with the metrics

  - The largest gaps (spearman, top‑3, candidate_valid_frac, semidense visibility) are exactly what you’d expect when you remove candidate‑specific cues and
    replace them with global/pose‑only features.
  - The visibility drop is consistent with the larger projection grid and different weighting, which then cascades into lower candidate_valid_frac and lower
    coverage‑weighted signals.
  - The model then effectively “sees” very similar features across candidates, so it collapses to a single class (matching the confusion matrices).

  If you want, I can produce a short “architecture diff checklist” that maps each missing module to a specific metric impact and to an action you can take
  without re‑introducing PointNeXt.


› please proceed as suggested!


• Here’s the architecture diff checklist you asked for — each item maps the v3 vs T41 diff to the observed metric deviations and a streamlined action (no
  PointNeXt).

  Architecture Diff Checklist (v3 vs T41)

  - Semidense projection features are not fed into the head (v3) → T41 concatenates semidense_proj (and frustum/point/traj features) to the head; v3 drops
    all of that. This directly reduces candidate‑specific signal and matches the big Spearman/top‑3 collapse.
      - Evidence: train-aux/spearman 0.024 vs 0.328; top3_accuracy 0.201 vs 0.285.
      - Streamlined action: append semidense_proj (5 dims) to head input; if you want more capacity, pass it through a tiny MLP (e.g., 5→16) before
        concatenation.
  - Semidense frustum MHCA removed (v3) → T41 uses candidate‑conditioned attention over projected points (enable_semidense_frustum=True) which helps
    discriminate candidates. Removing it aligns with mode collapse.
      - Evidence: loss +0.39/+0.41; Spearman drop.
      - Streamlined action: add a frustum‑lite summary (e.g., per‑candidate depth quantiles or histogram bins) derived from the same projection data — no
        attention module.
  - Trajectory context removed (v3) → T41 uses use_traj_encoder=True and attention against pose queries. Losing temporal context reduces rankability.
      - Evidence: same Spearman/top‑3 gap; T41 consistently higher.
      - Streamlined action: add a cheap trajectory feature: mean of encoded rig poses (or last‑first delta) and concatenate to head; avoid full attention.
  - Voxel validity handled as a gate (v3) vs feature (T41) → T41 uses use_voxel_valid_frac_feature=True and use_voxel_valid_frac_gate=False. v3 gates global
    features, which can suppress gradients for many candidates.
      - Evidence: candidate_valid_frac 0.78 vs 1.00 (train/val); coverage_weight_mean lower.
      - Streamlined action: disable gating; append [voxel_valid_frac, 1-voxel_valid_frac] to head features.
  - Semidense projection resolution differs (semidense_proj_grid_size=16 in v3 vs 12 in T41) → finer grid lowers coverage, which reduces
    semidense_candidate_vis_frac and thus candidate validity.
      - Evidence: semidense_candidate_vis_frac_mean 0.0619 vs 0.0823.
      - Streamlined action: set grid size to 12 to match sweep best.
  - Semidense weighting strategy differs → v3 always applies obs_count + inv_dist_std weighting; T41 used semidense_obs_count_norm=none. That can depress
    visibility and stability.
      - Evidence: lower semidense visibility + candidate_valid_frac in v3.
      - Streamlined action: add a “no‑weighting” mode (or set weights=1) to match T41 behavior.
  - Head depth & capacity mismatch (head_num_layers=3, field_dim=16 in v3 vs head_num_layers=1, field_dim=24 in T41). With reduced inputs, a deeper MLP can
    overfit or stall.
      - Evidence: higher losses and weak rank signal.
      - Streamlined action: set head layers to 1 and field_dim to 24 (or keep 16 and reduce MLP depth).
  - Projection‑based modulation changed → T41 uses sem_proj_film to modulate global features with candidate‑specific semidense stats; v3 uses
    voxel_proj_film from pooled voxel centers, which is less candidate‑specific.
      - Evidence: lower candidate_valid_frac and lower semidense visibility.
      - Streamlined action: keep voxel_proj_film (per your request), but also inject semidense_proj into the head, which restores candidate‑specific signal
        without re‑introducing FiLM on semidense.

  If you want, I can turn this into a concrete “minimal v3.1 baseline” proposal (specific fields to add and exact config values) while keeping PointNeXt
  out.


› please enrich docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ with these findings and remove the table that is currently present.


• Updated docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ to remove the top‑trials table and add the vin‑v3‑01 vs T41 deviation analysis plus
  architecture‑delta evidence, so the appendix now directly explains the observed mode‑collapse without relying on the removed figure.

  - docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ now has three new subsections: observed deviations (with concrete metric deltas),
    architecture deltas tied to those metrics, and streamlined adjustments that keep PointNeXt out.
  - Rewrote the sweep summary sentence to avoid the removed table reference while keeping the CSV‑derived counts in the narrative.
  - Added explicit run IDs and metric names in monospace for traceability.

  Tests

  - typst compile --root docs docs/typst/paper/main.typ


› Ok, what changes to oracle_rri/oracle_rri/vin/model_v3.py do you suggest. I just started a training with the current .configs/offline_only.toml, and
  again in the first epoch it shows mode-collapse, which was not the case for T41.Also the training dymaics don't show the same downward slope in the
  train-loss!


• You’re seeing immediate mode‑collapse and flat train loss with the current v3 + .configs/offline_only.toml, so you want concrete model_v3.py changes that
  restore candidate‑specific signal without re‑introducing PointNeXt.

  Highest‑impact fixes (keep v3 streamlined)

  - Feed semidense projection stats into the head.
    Right now v3 computes semidense_proj but never concatenates it into the scorer input. That removes the strongest candidate‑specific cue and matches the
    spearman/top‑3 collapse.
      - Change: append semidense_proj (5 dims) to head input, or pass it through a tiny MLP (e.g., 5→16) before concatenation.
  - Prefer a raw visibility fraction for candidate validity + loss weighting.
    v3 uses a weighted visibility fraction (with n_obs and 1/σ_d weighting), which lowers semidense_candidate_vis_frac and candidate_valid_frac. That
    directly matches the observed gap.
      - Change: compute both raw and weighted vis; use raw for candidate_valid and coverage weighting; keep weighted for logging/diagnostics.

  Secondary but likely important

  - Disable voxel gating by default and add voxel validity as features instead.
    v3 currently multiplies global_feat by a gate (hard suppression). T41 kept the gate off and used validity as features; this is more forgiving early in
    training.
      - Change: default use_voxel_valid_frac_gate=False, append [v, 1−v] to head input.
  - Align projection grid & pooling defaults to the best sweep.
    semidense_proj_grid_size=16 vs T41’s 12 lowers visibility and increases invalid candidates.
      - Change: default grid size to 12; global pool grid size to 5.
  - Reduce head depth.
    Your v3 run used head_num_layers=3 vs T41’s 1. With weaker inputs, deeper MLPs tend to collapse.
      - Change: default head_num_layers=1, keep head_hidden_dim=192.

  Optional low‑cost candidate signal (no attention)

  - Add a tiny trajectory summary feature.
    Without full MHCA, you can use mean/last trajectory pose encoding (single vector) concatenated to the head. It’s cheap and preserves some temporal
    context.

  Fail‑fast additions (to prevent silent collapse)

  - Add a runtime check in _forward_impl that raises or logs if semidense_proj is near‑constant across candidates (e.g., low std) or if candidate_valid_frac
    is below a threshold for most batches.

  If you want me to implement, I can do this as a minimal patch set:

  1. semidense_proj → head input (tiny MLP optional)
  2. raw vs weighted visibility split
  3. default config tweaks (grid size, gating, head depth)

  Just say the word and I’ll apply these changes.


› T41 had a much shorter one-cycle lr - hence it had a much higher learning rate early on - inspect lr-AdamW of both runs, as wells as important train-
  gradnorms/**! Include analysis of the objective w.r.t. observatiosn in the grad-norms and lr to docs/typst/paper/sections/12g-appendix-vin-v3-
  streamline.typ!


• Added LR/grad‑norm diagnostics to the VINv3 streamline appendix and removed the old sweep table, linking the mode‑collapse to the one‑cycle schedule and
  gradient energy differences. This is now captured in docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ.

  Key findings inserted into the appendix:

  - lr‑AdamW: v3 peaks later/lower (6.61e‑5 at step 353) vs T41 peak 1.83e‑4 at step 32 with full decay by step 266.
  - Early grad‑norm means are ~6–16× smaller in v3 for core modules (head MLP, head CORAL, global pooler, field proj), while T41 has large gradients in
    candidate‑specific paths (traj/frustum/point encoder) that v3 lacks.

  Files touched:

  - docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ
  - .codex/vin_v3_lr_gradnorm_appendix.md (task notes)

  Build check:

  - typst compile --root docs docs/typst/paper/main.typ

  If you want the raw W&B summary numbers dumped into a CSV or a small inline table, I can add that next.


› What do you think is the likeliest cause for the mode-collapse we observed so far when training vin-v3?


• Most likely: the head in v3 is effectively blind to candidate‑specific evidence, so it learns a near‑constant predictor. In v3 the semidense projection
  stats are computed but not fed into the scorer, and frustum/trajectory/point‑encoder paths are gone. That removes the strongest per‑candidate signal T41
  used. On top of that, visibility weighting + gating reduces candidate_valid_frac, so the remaining gradients are weak and uniform. The W&B evidence fits
  this: v3’s early grad norms are an order of magnitude smaller and the LR peaks later/lower, which amplifies the collapse rather than rescuing it.

  If I had to pick a single cause: “candidate‑specific features never reach the head.” The LR/grad‑norm differences look like secondary amplifiers of that
  architectural bottleneck.


› Elaborate! What do you mean by semidense-projection stats not being fed to the scorer? we're now projecting the semi-dense features into each candidates
  frame, right?


• Yes, you are projecting semidense points into each candidate frame — but in v3 those projection summaries never reach the scoring MLP, so they only
  influence candidate_valid and logging.

  Concretely in oracle_rri/oracle_rri/vin/model_v3.py:

  1. Projection happens (candidate‑specific)
      - _sample_semidense_points(...)
      - _project_semidense_points(...)
      - _encode_semidense_projection_features(...) → semidense_proj (shape B × Nq × 5)
  2. But the scorer input ignores them
     The head is built from only:

     parts = [
         pose_feats.pose_enc,  # candidate pose encoding
         global_feat,          # pooled voxel field (mostly scene‑level)
     ]
     feats = torch.cat(parts, dim=-1)
     logits = head_coral(head_mlp(feats))

     So the candidate‑specific semidense stats do not affect logits.
  3. Where semidense_proj is used now
     It only drives:

     semidense_candidate_vis_frac = semidense_proj[..., idx]
     candidate_valid = pose_finite & (voxel_valid_frac > 0) & (semidense_candidate_vis_frac > 0)

     i.e., it masks or gates, but doesn’t provide discriminative input.

  ———

  ### Why this causes mode‑collapse

  - With only pose encoding + global voxel features, candidates become hard to distinguish.
  - The head learns that predicting a constant class minimizes loss (especially with weak gradients).
  - The strong candidate‑specific cue (how much semidense evidence is visible from each view) never reaches the head, so it can’t rank candidates properly.

  ———

  ### What it should look like (minimum fix)

  Add semidense projection stats to the head input:

  parts = [
      pose_feats.pose_enc,
      global_feat,
      semidense_proj,  # candidate‑specific visibility/coverage signal
  ]

  Optionally: pass semidense_proj through a tiny MLP (5→16) before concatenation.

  ———

  ### Bottom line

  You’re computing the right candidate‑specific semidense info, but only using it for validity filtering. The scorer never sees it, so it can’t learn to
  differentiate candidates → mode‑collapse.

  If you want, I can point to the exact line numbers in model_v3.py where semidense_proj is computed and then dropped.


› Fix this, they should be included in parts!


• Updated oracle_rri/oracle_rri/vin/model_v3.py so semidense projection stats are actually fed into the scorer and the head input dimension matches. This
  restores candidate‑specific signal instead of only using it for validity masking.

  - Added SEMIDENSE_PROJ_DIM to head_in_dim and appended semidense_proj to parts in _forward_impl.
  - Left all other pathways unchanged.

  Tests run:

  - ruff format oracle_rri/oracle_rri/vin/model_v3.py
  - ruff check oracle_rri/oracle_rri/vin/model_v3.py
  - uv run pytest oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/vin/test_vin_model_v3_methods.py

  Notes: logged the change in .codex/vin_v3_semidense_head.md.

  If you want, I can also add a tiny MLP for semidense_proj (5→16) to give it a bit more capacity without reintroducing heavy modules.


› We're currently using both a sigmoid gate, as well as FiLM to modulate the voxel features. Please investigate. Only keep one.


• Kept only one modulation path by removing the voxel gate and enforcing FiLM‑only modulation. use_voxel_valid_frac_gate=True now raises to avoid silent
  dual‑modulation.

  Changes

  - Removed voxel gate creation + application in oracle_rri/oracle_rri/vin/model_v3.py.
  - Set use_voxel_valid_frac_gate default to False and mark as deprecated; raises if enabled.
  - Updated test config to disable the gate.
  - Logged the change in .codex/ and updated .codex/AGENTS_INTERNAL_DB.md.

  Files touched

  - oracle_rri/oracle_rri/vin/model_v3.py
  - oracle_rri/tests/vin/test_vin_model_v3_methods.py
  - .codex/vin_v3_voxel_gate.md
  - .codex/AGENTS_INTERNAL_DB.md

  Tests

  - ruff format oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/tests/vin/test_vin_model_v3_methods.py
  - ruff check oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/tests/vin/test_vin_model_v3_methods.py
  - uv run pytest oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/vin/test_vin_model_v3_methods.py

  Note: your .configs/offline_only.toml must set use_voxel_valid_frac_gate=false now, otherwise VINv3 will raise at init. If you want me to patch the
  config, say the word.


› Please use $aria-nbv-context to figure out, which features the original VIN-NBV paper (VIN-NBV-frahm2025) computed and provide an overview of the
  differences to our oracle_rri/oracle_rri/vin/model_v3.py!


• Using $aria-nbv-context I pulled the VIN‑NBV (Frahm 2025) feature description from literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex. Here’s the
  distilled feature set and how it differs from oracle_rri/oracle_rri/vin/model_v3.py.

  VIN‑NBV (Frahm 2025) feature pipeline (from sec/3_methods.tex)

  - Reconstruction: build point cloud ( \mathcal{R}_{base} ) from base RGB‑D + camera params.
  - Per‑point enrichment: surface normals, visibility count (# views that saw the point), and depth.
  - Candidate projection: project enriched cloud into each candidate view, forming a 512×512×5 feature grid (normals + visibility count + depth).
  - Downsample + variance: pool to 256×256, compute per‑pixel variance (captures local geometric complexity).
      - (F_p) = downsampled per‑pixel grid
      - (F_v) = per‑pixel variance grid
  - Coverage proxy: (F_{empty}) is a 2‑element vector counting empty pixels inside and outside the hull of non‑empty pixels.
  - Acquisition stage cue: (F_{base}) = number of base views.
  - Encoder + MLP: CNN ( \mathcal{E}\theta(F_v \oplus F_p) \Rightarrow V{view} \in \mathbb{R}^{256} ), then
    ( \widehat{RRI}(q) = \mathcal{M}\phi(V{view} \oplus F_{empty} \oplus F_{base}) ).

  VINv3 (ours) feature pipeline

  - Scene field: EVL voxel field channels (e.g., occ_pr, occ_input, counts_norm, cent_pr, free_input, new_surface_prior).
  - Pose encoding: R6D + LFF for candidate poses (explicit pose conditioning).
  - Global context: Pose‑conditioned global pooling over the voxel field.
  - Semidense projection stats: from semidense points, per‑candidate coverage, empty_frac, visibility, depth mean, depth std (5‑D).
  - FiLM: voxel‑projection FiLM modulates global features using candidate‑specific stats.
  - Head input: pose_enc + global_feat + semidense_proj (as of the latest fix).

  Key differences (VIN‑NBV vs VINv3)

  - 2D grid vs 3D voxel field:
    VIN‑NBV encodes a dense 2D projected grid + variance; v3 uses a 3D voxel field with global pooling.
  - Normals/variance vs coverage stats:
    VIN‑NBV explicitly uses surface normals and per‑pixel variance as local geometry complexity; v3 uses coverage/visibility + depth stats from semidense
    projections (no normals, no pixel‑variance map).
  - Coverage proxy:
    VIN‑NBV’s (F_{empty}) counts empty pixels inside/outside a hull. v3 uses coverage/empty_frac from projection bins (different geometry prior).
  - Stage cue:
    VIN‑NBV includes (F_{base}) (number of base views). v3 does not include a base‑view count feature.
  - Encoder:
    VIN‑NBV uses a CNN encoder on projected 2D grids; v3 uses global pooling + MLP (no 2D conv).
  - Candidate‑specific signal path:
    VIN‑NBV’s per‑candidate projected grid is the primary input to the scorer. v3 uses semidense projection stats as a compact candidate signal (now
    injected into the head).

  If you want, I can add a short side‑by‑side table into the appendix (or a paper section) with these feature deltas and cite the exact lines from the
  VIN‑NBV LaTeX sources.


• Here’s how those coverage‑weighting options are implemented in oracle_rri/oracle_rri/lightning/lit_module.py, and what they actually do during training.

  Where the logic lives

  - Coverage weighting is computed in _coverage_weights() and applied in _step() (train only).
  - It is not applied in val/test (_coverage_weights returns None for non‑TRAIN stages).
  - Metrics logged: Metric.COVERAGE_WEIGHT_MEAN and Metric.COVERAGE_WEIGHT_STRENGTH.

  ———

  What each option controls

  - coverage_weight_mode = "voxel"
      - Chooses which coverage fraction to use as the per‑candidate weight source.
      - Implemented in _select_coverage_fraction().
      - "voxel" uses pred.voxel_valid_frac (from VIN forward).
      - Other valid modes:
          - "semidense" → pred.semidense_candidate_vis_frac (or semidense_valid_frac)
          - "min", "mean", "product" → combine voxel + semidense
          - "none" → disables weighting entirely.
  - coverage_weight_floor = 0.2
      - Minimum loss weight for low‑coverage candidates.
      - The base weight is:

        base_weight = floor + (1 - floor) * coverage^power
      - With floor=0.2, no candidate gets <0.2 weight, even if coverage is near 0.
  - coverage_weight_power = 1.0
      - Exponent applied to the coverage fraction before weighting.
      - 1.0 keeps it linear;
        >1 emphasizes high‑coverage samples;
        <1 flattens differences.
  - coverage_weight_strength_start = 0.6
      - Initial blend strength between uniform weighting and coverage weighting.
      - Final weights are:

        weights = lerp(1.0, base_weight, strength)
      - With 0.6, you start at 60% coverage‑based weighting + 40% uniform.
  - coverage_weight_strength_end = 0.0
      - Target strength after annealing ends.
      - 0.0 means you fully disable coverage‑weighting after anneal completes (uniform weights).
  - coverage_weight_schedule = "linear"
      - How strength transitions from start → end.
      - Implemented in _coverage_weight_strength():
          - "linear": straight interpolation
          - "cosine": smooth cosine decay
  - coverage_weight_interval = "epoch"
      - Whether the anneal schedule is driven by epoch or global step.
      - "epoch" means strength changes once per epoch (using self.current_epoch).
  - coverage_weight_anneal_epochs = 3
      - Total epochs to linearly anneal from strength_start to strength_end.
      - With your settings:
          - epoch 0 → strength ≈ 0.6
          - epoch 1 → ~0.4
          - epoch 2 → ~0.2
          - epoch 3+ → 0.0 (uniform weights)
  - coverage_weight_apply_aux = true
      - Applies the same coverage weights to the auxiliary regression loss (if enabled).
      - In _step():

        aux_loss = (aux_loss_per * coverage_weights).sum() / coverage_weights.sum()
      - If aux_regression_loss is None, this flag has no effect.

  ———

  End‑to‑end effect with your settings

  - Training only.
  - Uses voxel coverage fraction (pred.voxel_valid_frac) as the weighting signal.
  - Early epochs: candidates with higher voxel coverage get higher loss weight, but no one drops below 0.2.
  - Over 3 epochs, the weighting fades to uniform (strength_end=0.0).
  - Both CORAL loss and aux regression loss are weighted.

  ———

  If you want, I can also map these to specific W&B metrics (e.g., how the logged coverage_weight_mean changes per epoch with these settings).


› Checkout the current state of our typst paper using [$aria-nbv-context](/home/jandu/repos/NBV/.codex/skills/aria-nbv-context/SKILL.md); then provide a
  concise scientific appendix section on the scheduled coverage reweighting, what's our motivation to apply it - curriculum learning, high variance in
  gradient signal, ...


• I checked the current Typst paper structure (via scripts/nbv_typst_includes.py) and the relevant existing mentions are in docs/typst/paper/sections/07b-
  training-config.typ, docs/typst/paper/sections/09b-ablation.typ, and docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ. Below is a concise
  appendix subsection you can drop into the appendix (I’d place it in 12g-appendix-vin-v3-streamline.typ or 12b-appendix-extra.typ). It follows the paper’s
  style and mirrors the actual implementation in lit_module.py.

  == Scheduled coverage reweighting (curriculum)

  #import "../../shared/macros.typ": *

  We apply a scheduled coverage reweighting only during training to stabilize
  early optimization when candidate evidence is sparse or noisy. Each candidate
  $i$ provides a coverage fraction $c_i \in [0,1]$ (e.g., voxel-valid fraction
  or semidense visibility), which we map to a base weight

  $
    \tilde{w}_i = f + (1 - f)\, c_i^p
  $

  with floor $f$ and exponent $p$ to avoid zeroing low-coverage samples while
  allowing sharper emphasis on high-evidence candidates. The actual weight is
  a blend between uniform weighting and coverage weighting:

  $
    w_i(t) = (1 - \lambda_t) + \lambda_t\, \tilde{w}_i,
  $

  where $\lambda_t$ is an annealed strength schedule (linear or cosine) from
  $\lambda_0$ to $\lambda_T$ over a fixed number of epochs (or steps). The loss
  is then aggregated as a weighted average,

  $
    cal(L) = \frac{\sum_i w_i(t)\, \ell_i}{\sum_i w_i(t)},
  $

  and the same weights optionally apply to the auxiliary regression term.

  *Motivation.* Early in training, low-coverage candidates often have ambiguous
  geometry and noisy RRI labels, which increases gradient variance and encourages
  collapse toward the dominant ordinal bins. The schedule acts as a curriculum:
  it prioritizes high-evidence candidates to establish a stable ranking signal,
  then anneals back toward uniform weighting so the model still learns from
  difficult, low-coverage cases. The floor $f$ guarantees that all candidates
  retain a nonzero gradient contribution throughout the schedule.

  If you want, I can insert this directly into docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ (or another appendix file) and recompile.


› please proceed


• Added the appendix subsection and citation, and recompiled successfully.

  What changed

  - Added a new appendix subsection on scheduled coverage reweighting (curriculum motivation + equations) in docs/typst/paper/sections/12g-appendix-vin-v3-
    streamline.typ.
  - Added curriculum learning citation to docs/references.bib.

  Compile

  - typst compile --root docs docs/typst/paper/main.typ

  Notes

  - I used Typst math syntax (lambda without backslash) to avoid compile errors.

  If you want tweaks to the wording or placement (e.g., move into 07b-training-config.typ instead), tell me where to shift it.


› Do we differentiate between semi-dense points based on their visibility from each candidate view?


• Yes—per candidate view we compute a visibility mask and use it to summarize semidense points.

  In v3 we project semidense points into each candidate camera and compute valid = finite & (z>0) & in-bounds. That mask is per‑candidate (since projection
  uses the candidate’s p3d_cameras). It drives:

  - coverage/empty/visibility fractions and depth stats in _encode_semidense_projection_features, and
  - the occupancy/depth grids used by the semidense CNN in _encode_semidense_grid_features.

  So the model differentiates points only via this candidate‑specific visibility mask (plus optional reliability weights from inv_dist_std/obs_count). There
  is no explicit per‑point visibility embedding and no mesh‑occlusion visibility—it’s purely frustum/in‑bounds visibility.

  Refs: oracle_rri/oracle_rri/vin/model_v3.py (_project_semidense_points, _encode_semidense_projection_features, _encode_semidense_grid_features).


› how do we use valid?


• Short answer: valid is the per‑candidate projection mask and it drives all semidense projection statistics (and the CNN grid), plus the candidate‑validity
  flag. It does not explicitly drop points beyond those summaries, and it doesn’t model occlusion—just in‑bounds + positive depth + finite.

  Where it’s used (v3):

  - _project_semidense_points computes
    valid = finite & (z>0) & in-bounds per candidate view (B·Nq × P).
  - _encode_semidense_projection_features uses valid to:
      - build the coverage grid via scatter_add (valid points only),
      - compute coverage / empty_frac,
      - compute semidense_candidate_vis_frac as a weighted valid/finite fraction,
      - compute depth mean/std (only valid points contribute).
  - _encode_semidense_grid_features uses valid to:
      - bin points into the G×G grid and accumulate occupancy, depth mean, depth std,
      - invalid points contribute zero.
  - _forward_impl uses the resulting semidense_candidate_vis_frac to build
    candidate_valid = pose_finite & voxel_valid_frac>0 & semidense_candidate_vis_frac>0.

  Training implications:

  - Coverage weighting (if enabled) uses semidense_candidate_vis_frac and/or voxel_valid_frac, which both depend on valid.
  - Loss masking in lit_module does not use candidate_valid; it’s only logged.

  So valid is the core mask for all candidate‑specific semidense stats and the CNN grid, and it indirectly affects coverage‑weighted loss via those stats.


› Please make use of $aria-nbv-context and potentially $typst-authoring and revise docs/typst/paper/sections/06-architecture.typ such that all
  implementation details - i.e. per-candidate encodings of the semi-dense pointcloud and all other relevant computations and transforms are described
  concisely and academically!


• Updated the architecture section to match the current VINv3 implementation, with explicit per‑candidate semidense projections, voxel‑projection FiLM,
  validity logic, and optional trajectory context. The text now aligns with model_v3.py and removes v2‑only components (frustum MHCA, PointNeXt), while
  marking the v2 diagram as legacy reference.

  Changes

  - docs/typst/paper/sections/06-architecture.typ: rewrote core/optional components, added voxel‑projection FiLM section, detailed semidense projection +
    validity mask + reliability weighting, added grid‑CNN description, updated feature fusion vector.
  - docs/references.bib: unchanged (FiLM citation already present).

  Compile

  - typst compile --root docs docs/typst/paper/main.typ

  Notes

  - New appendix entry recorded in .codex/architecture_v3_update.md.

  If you want me to tighten any of the math notation or add a brief pseudo‑algorithm box for the per‑candidate projection pipeline, I can do that next.


• Updated the W&B analysis utilities and pulled the latest run’s dynamics via the API. Here’s the analysis and the code changes.

  Changes made

  - Moved non‑UI W&B analysis helpers into oracle_rri/oracle_rri/configs/wandb_config.py (new helpers: _ensure_wandb_api, _get_run, _load_runs_filtered,
    _run_metadata, _resolve_x_key, _summarize_metric, _summarize_gap, _load_wandb_history_clean, plus slope/segment helpers).
  - Simplified oracle_rri/oracle_rri/app/panels/wandb.py to reuse those utilities and removed duplicated logic.

  Analysis: run rtjvfyyp (v03-best)
  Key dynamics (train/val coral_loss_rel_random):

  - Epoch count: epoch=25 in history/summary. No early‑stopping keys were logged in the run config/summary; stopping is most consistent with max_epochs=25.
    If you saw 22, likely a logging/zero‑index confusion or different run.
  - Train loss (epoch): early mean ≈ 0.742 → mid ≈ 0.673 → late ≈ 0.636; late slope ≈ −3.5e‑05 per step (very shallow).
  - Val loss (epoch): early mean ≈ 0.725 → mid ≈ 0.676 → late ≈ 0.665; late slope ≈ −6.1e‑06 per step (essentially flat).
  - LR (lr-AdamW): decays monotonically from 8e‑4 → 4.34e‑4. Peak is at step 0 (no warmup rise).
  - LR vs step‑level loss: corr ≈ +0.43 (higher LR correlates with higher loss).
  - Step‑level loss noise (late segment): CV ≈ 0.11, min ≈ 0.45, max ≈ 0.91.
  - Grad‑norms (late CV): ~0.33–0.56 across modules (global_pooler, field_proj, traj_*, head_*) → moderate noise.

  Last 3 finished runs in project (for context)

  - rtjvfyyp is the best of the latest three by both train/val loss; the other two are substantially higher and flatter.

  Answers to your questions

  - Higher batch size?
      - Evidence: step‑loss CV ≈ 0.11 and grad‑norm CV ~0.33–0.56 suggests moderate stochasticity. Larger batch (or gradient accumulation) should reduce
        noise and might stabilize the objective, but expect diminishing returns. If you increase batch size, consider LR scaling (linear or sqrt) to keep
        effective step size comparable.
  - Drop OneCycleLR?
      - In this run, the “one‑cycle” behaves like a mild monotonic decay (8e‑4 → 4.3e‑4) with no warmup peak. Loss correlates positively with LR and
        continues to improve slowly as LR drops. This suggests the schedule is not catastrophic, but it’s also not giving you the early high‑LR push seen in
        T41.
      - Recommendation: either (a) keep OneCycleLR but set a real warmup (pct_start) and a higher max_lr to mimic T41, or (b) switch to constant LR + manual
        decay or ReduceLROnPlateau to let the objective settle further once it plateaus. I would not drop OneCycleLR purely on this run; the problem is
        likely the parameterization.
  - Why did training stop after 22 epochs?
      - W&B shows epoch=25 and no early‑stop signals in the run config/summary. Most likely the run hit max_epochs=25. If you’re seeing 22 locally, check
        the trainer config used in that launch (and whether you’re logging 0‑indexed epochs).
  - Do the dynamics suggest further improvement beyond 25 epochs?
      - Late slopes are ~0 (train −3.5e‑05, val −6.1e‑06 per step). This is a plateau signal: small gains possible, but without changing LR schedule (or
        increasing batch size/regularization), improvements will likely be marginal.
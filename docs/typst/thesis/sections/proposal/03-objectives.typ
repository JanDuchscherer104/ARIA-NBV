#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Objectives and Hypotheses

The proposal has one primary objective: establish a reproducible, target-aware,
finite-candidate #NBV stack whose selected views improve target reconstruction
quality under a fixed acquisition budget. The work is evidence-gated so that a
failed substrate does not become an overstated policy claim.

*O1: Trustworthy geometry and oracle labels.* M1 must demonstrate that offline
store indexing, splits, frame conventions, CW90 display handling,
candidate-label order, depth rendering/backprojection, invalidity reasons, and
Rerun inspection agree on a small trusted subset. The output is not merely a
smoke test; it is a public contract that says when #RRI labels can be used for
training and scale-up.

*O2: Target-specific utility without actor leakage.* For selected target $e$,
the thesis must compute scene #RRI and target #RRI separately, using
actor-visible observed/predicted target descriptors $z_e$ for selection and
model input, but GT crops $M_e$ for labels/evaluation. The first descriptor is
predicted/observed OBB center, extents, orientation, class, confidence,
projected area, semidense support, EVL support, and relative pose. Crop
descriptors and entity tokens are ablations, not initial dependencies.

*O3: A learned myopic control.* The target-conditioned one-step scorer is the
required comparator for planning. It should predict target #RRI over the same
candidate table and report held-out rank correlation, top-$k$ oracle hit rate,
calibration, target visibility, invalid fraction, and failure groups by
occlusion, small target, weak descriptor, and candidate infeasibility.

*O4: Replayable multi-step data.* Rollout data must include random-valid,
oracle-greedy/lookahead, and oracle-scored temperature-softmax traces with
deterministic replay metadata. Gumbel-Top-k is preferred later diversity
evidence, but deterministic greedy/lookahead must be trusted before stochastic
branching becomes training data.

*O5: Mandatory finite-candidate $Q_H$.* The hard thesis result is a masked
candidate-query Transformer trained from #ASE oracle rollout traces. It emits
one value per candidate:

$ bold(h)_(t,i)
  = f_theta(
      x_"scene"(s_t^"cf0"),
      x_"target"(z_e),
      x_"hist"(a_(<t), r_(<t)),
      x_"cand"(q_(t,i))
    ), quad
  Q_(H,theta)(s_t^"cf0", z_e, q_(t,i)) = W_Q bold(h)_(t,i). $

The selected action is

$ a_t^theta
  = op("argmax", limits: #true)_(i: m_(t,i)=1)
    Q_(H,theta)(s_t^"cf0", z_e, q_(t,i)). $

Success is measured after oracle re-evaluation of the selected actions, not by
the network's self-reported value. $Q_H$ must beat one-step greedy/model
scoring on cumulative target #RRI under equal acquisition and candidate budgets,
with bounded oracle lookahead reported as the upper bound. If it does not, the
thesis reports the failing gate and the smallest defensible target-aware result.

#figure(
  table(
    columns: (0.82fr, 1.22fr, 1.7fr),
    table.header([*Claim*], [*Metric*], [*Evidence rule*]),
    [Target utility],
    [$J_e^(H)$, $G_t^(H)$, scene #RRI, log-gain ablation],
    [Endpoint target gain and cumulative return are reported separately; cost curves compare different budgets.],
    [Target input safety],
    [selector rank, OBB match score, target support, leakage checks],
    [V1 actor input uses observed/predicted evidence; GT target crops only produce labels/evaluation.],
    [Candidate space],
    [valid fraction, invalid reasons, target visibility, sampler provenance],
    [All candidates carry hard masks and strategy provenance before scorer or $Q_H$ training.],
    [Planning gain],
    [oracle-evaluated cumulative target #RRI under equal budget],
    [$Q_H$ is compared to random-valid, one-step greedy, learned one-step target scorer, and oracle lookahead.],
    [Scale],
    [scenes, snippets, targets, trajectories, seeds, transitions, gaps],
    [Final scale targets all 100 GT-mesh #ASE scenes and 4,608 snippet windows, or an exact scene-level held-out subset report.],
  ),
  caption: [Objective-to-evidence matrix.],
) <tab:proposal-objective-evidence>

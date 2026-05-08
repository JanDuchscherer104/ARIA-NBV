# Thesis Prose Fixture

Use this as a mental regression test when polishing ARIA-NBV writing.

## Bad

This work leverages semantic relevance to unlock a holistic and powerful
next-best-view pipeline for AR reconstruction. The proposed method highlights
the crucial role of entity awareness in the rapidly evolving landscape of
interactive 3D reconstruction.

## Better

ARIA-NBV scores candidate views by expected reconstruction improvement for the
selected entity, not only by scene-level surface coverage. This changes the NBV
objective from "what improves the global model most?" to "what improves the
user-relevant target under the current interaction context?"

## Why Better

- It names the mechanism: candidate scoring by entity-specific expected
  improvement.
- It states the contrast: entity-level versus scene-level objective.
- It avoids unsupported importance claims.
- It gives the reader a testable interpretation of the method.

## Contribution Paragraph

Bad:

> The thesis introduces a robust and comprehensive framework for intelligent
> AR guidance.

Better:

> The thesis tests a finite-candidate ARIA-NBV stack in which target-specific
> oracle RRI supervises a target-conditioned scorer and, after an oracle
> headroom check, a masked finite-horizon `Q_H` model.

## Related-Work Contrast

Bad:

> Many NBV methods exist, but ARIA-NBV is more semantic and practical.

Better:

> Learned NBV methods provide policies or scorers for view selection, but the
> ARIA-NBV thesis isolates a narrower question: whether target-conditioned
> reconstruction-quality labels improve finite candidate selection in logged
> egocentric AR snippets.

## Method Overview

Bad:

> The system uses several modules to process data and make decisions.

Better:

> At decision time, the actor observes logged geometry, frozen egocentric
> features, predicted target descriptors, and a finite valid candidate set;
> ASE meshes and GT target crops remain oracle-only assets for labels and
> endpoint evaluation.

## Limitation Paragraph

Bad:

> Future work will extend the method to real devices and continuous control.

Better:

> The thesis does not establish real-device deployment or continuous-control
> performance. Those extensions require an online reward/evaluation loop beyond
> the offline ASE mesh-oracle substrate used for the core experiments.

## Result Paragraph

Bad:

> The results are promising and demonstrate the strength of the approach.

Better:

> A result paragraph should name the metric, split, baseline, effect size, and
> uncertainty. If those numbers are not available, phrase the sentence as an
> experimental objective or hypothesis rather than an empirical conclusion.

## Caption Paragraph

Bad:

> Overview of our pipeline.

Better:

> Target-conditioned oracle supervision pipeline. Actor-visible observations
> define the target descriptor and candidate set, while GT mesh crops are used
> only to compute target-specific RRI labels and endpoint metrics.

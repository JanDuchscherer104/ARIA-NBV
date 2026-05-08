# ARIA-NBV Thesis Writing Guide

This is calibrated for a computer-science master thesis on target-conditioned
next-best-view planning, not for biomedical journal submission.

## Proposal And Thesis Rhetoric

Use a compact ABT/CARS structure:

- AND / territory: active 3D reconstruction, NBV planning, AR guidance,
  semantic scene representations, and learned reconstruction proxies.
- BUT / niche: handheld AR needs target-aware view choice, efficient
  counterfactual candidate evaluation, and leakage-safe oracle supervision.
- THEREFORE / contribution: ARIA-NBV proposes a target-conditioned,
  quality-driven finite-candidate NBV stack and validates it under fixed
  acquisition budgets.

The gap must follow from the context. Do not bolt on "semantic relevance" as a
slogan; state the concrete failure mode it addresses.

## Section Jobs

Introduction: establish why interactive NBV matters, state the limitation of
geometry-only or scene-level framing, introduce ARIA-NBV, and end with precise
contributions.

Related work: organize by intellectual dependency, not a shopping list:
classical NBV, learned NBV, reconstruction-quality proxies/oracle labels,
entity-aware reconstruction, and why the thesis combination remains
non-trivial.

Method/system design: specify input/output contracts, frame conventions,
symbol/equation references, learned vs deterministic components, online vs
offline parts, and expected failure modes.

Experiments: report dataset/split assumptions, candidate sampling, oracle
labels, baselines, ablations, metrics, aggregation, runtime constraints,
limitations, and threats to validity.

Discussion: separate what experiments show, what the design suggests, what is
speculative, and what future AR-in-the-loop studies must validate.

## Paragraph Unit Test

For each paragraph, ask:

1. What is the paragraph's single job?
2. What concrete claim does it make?
3. What evidence supports it?
4. Which term or symbol must stay consistent with the shared library?
5. Does the last sentence set up the next paragraph?

If the answer is unclear, rewrite before polishing.

## Style

Prefer sober, specific prose.

Bad:

> This work leverages semantic awareness to unlock a powerful and holistic AR reconstruction pipeline.

Good:

> ARIA-NBV scores candidate views by expected reconstruction improvement for the selected target, not only by scene-level surface coverage.

Avoid filler: crucial, pivotal, revolutionary, landscape, delve, holistic,
foster, underscore, pave the way, seamlessly.

## Outline-To-Prose Workflow

1. Write a bullet outline with claims, evidence, and citations.
2. Convert each bullet cluster into paragraphs.
3. Remove bullets from final manuscript sections unless the template requires
   them.
4. Replace generic fluency with mechanisms, metrics, comparisons, or
   limitations.

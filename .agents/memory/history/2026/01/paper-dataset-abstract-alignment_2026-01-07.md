---
id: 2026-01-07_paper-dataset-abstract-alignment_2026-01-07
date: 2026-01-07
title: "Paper Dataset Abstract Alignment 2026 01 07"
status: legacy-imported
topics: [dataset, abstract, alignment, 2026, 01]
source_legacy_path: ".codex/paper-dataset-abstract-alignment_2026-01-07.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Paper alignment notes (2026-01-07)

Scope of this pass: tighten **implementation-faithful** wording in the Typst paper, add missing **ASE/ATEK-EFM dataset facts** (incl. our GT-mesh subset stats + LUF/LUF conventions), and remove “code identifier” leakage from main sections (identifiers belong in appendices).

## Changes applied (Typst paper)

- `docs/typst/paper/main.typ`
  - Rewrote the abstract to be closer in tone/content to `docs/index.qmd` and the VIN-NBV framing:
    - explicitly motivates why coverage proxies can fail in cluttered indoor scenes,
    - states oracle supervision as **relative reduction** in **bidirectional point↔mesh surface error**,
    - separates baseline inputs vs view-conditioned cues (semi-dense projections) without listing code symbols.
  - Kept the IEEE-style constraints (single paragraph, no citations, expand acronyms on first mention).

- `docs/typst/paper/sections/01-introduction.typ`
  - Reduced repetitive VIN-NBV citations and removed a few “generic” bridging phrases to sound less templated.
  - Clarified that GT meshes apply to a supervised subset.

- `docs/typst/paper/sections/02-related-work.typ`
  - Fixed “semi-dense” spelling in one place (avoid “semidense” drift in main text).
  - Added Project Aria ecosystem context (tooling + MPS outputs) while keeping the focus on NBV-relevant primitives.
  - Removed the privacy-features sentence to keep scope aligned with the technical contribution.

- `docs/typst/paper/sections/04-dataset.typ`
  - Added the missing “paper-grade” dataset facts:
    - snippet definition (20 frames @ 10 Hz, stride 10 frames),
    - ASE public release version (v1.0) and ATEK WebDataset export version (v0.1.1),
    - the actual per-stream resolutions used in our pipeline snapshot (RGB 240×240, depth 240×240, SLAM 240×320),
    - clarified GT-mesh supervised subset statistics from the local snapshot: **100 scenes / 4,608 snippets (median 40, min 8, max 152)**.
  - Removed the confusing “1,641 meshes” claim from the main text (keeps the paper falsifiable and consistent with what we actually train/evaluate on locally).
  - Fixed section references by adding a label to the entity-aware section and linking via `@sec:...` instead of “Sec. X”.

- `docs/typst/paper/sections/05-coordinate-conventions.typ`
  - Made LUF explicit and unambiguous (axis directions) and added a short note to avoid mixing camera axes with image pixel axes.
  - Kept the “dashboard 90° yaw” note but clearly scoped it to visualization only (removed “Implementation note:” phrasing).

- `docs/typst/paper/sections/05-oracle-rri.typ`
  - Removed “Implementation note:” phrasing while keeping the unambiguous statement that we use mean squared point-to-triangle / triangle-to-point distances and report accuracy + completeness.

- `docs/typst/paper/sections/07a-binning.typ`
  - Clarified current baseline vs planned feature: baseline uses a single global quantile binner; stage-aware normalization is future work (still documented for completeness).

- `docs/typst/paper/sections/07-training-objective.typ`
  - Fixed the CORAL BCE expression to correctly apply the sum to both positive and negative terms.

- `docs/typst/paper/sections/02-related-work.typ`
  - Rewrote the “Ordinal regression for continuous targets” subsection to:
    - define #textit(ordinal) classification clearly (ordered bins; distance-aware errors),
    - explain why VIN-NBV-style NBV prefers ordinal over direct regression (ranking objective, outliers, stage effects),
    - justify CORAL over nominal cross-entropy / independent one-vs-rest reductions (rank consistency, cumulative probabilities).

- `docs/typst/paper/sections/06-architecture.typ`
  - Added an implementation-faithful justification for the 6D rotation (R6D) pose representation:
    - avoids discontinuities of Euler/axis--angle/quaternion parameterizations,
    - uses a continuous Euclidean representation with Gram--Schmidt-like decoding,
    - prefers 6D over the 5D continuous alternative for simplicity and comparable/better empirical behavior @zhou2019continuity.
  - Removed Streamlit-placeholder figures (the paper currently uses the auto-generated Graphviz figure).

- `docs/typst/paper/sections/09a-evaluation.typ`
  - Replaced brittle “Sec. III” with a stable label reference to problem formulation (`@sec:problem`).

## Changes applied (ASE dataset documentation)

- `docs/contents/ase_dataset.qmd`
  - Corrected/clarified dataset-scale claims (added citations for headline numbers; removed contradictory mesh-count statements).
  - Added a quick-reference coordinate conventions subsection (LUF camera coordinates; gravity-aligned world frame via Project Aria / MPS).
  - Added a concise ATEK-EFM snippet-format subsection (fixed-length snippet windows; typical frame counts/stride; common per-stream resizing).

## Verification

- Compiled successfully:
  - `typst compile --root docs docs/typst/paper/main.typ .codex/_render/paper.pdf`
- Rendered PNG pages for inspection:
    - `.codex/_render/paper-01.png`
    - `.codex/_render/paper-02.png`
    - `.codex/_render/paper-03.png`

## “AI-generation” markers to watch (remaining)

These are not “wrong”, but they read templated and are typical reviewer red flags:

- Overuse of “We …” sentence openers in consecutive sentences (some remain in technical sections).
- “This paper summarizes / provides a reproducible baseline …” appears multiple times across sections; consider keeping it once (intro) and deleting elsewhere.
- Several sections read like documentation rather than a paper (lots of module-level nouns without a crisp “what is new / what is evaluated” claim).

## Next suggested tightening pass (optional)

- Replace remaining “implementation note:” paragraphs in main sections with neutral scientific phrasing, unless they truly prevent ambiguity.
- Ensure “baseline vs optional vs planned” is consistently signposted in:
  - `docs/typst/paper/sections/06-architecture.typ`
  - `docs/typst/paper/sections/09b-ablation.typ`
- Decide whether stage-aware binning is (a) implemented now (then describe exactly), or (b) future work (then avoid presenting it as a current method).
 - If you want slightly more formalism, add a 2-line equation snippet for the 6D→SO(3) Gram–Schmidt-like decoding (still without code identifiers).

# Abstract/Intro/Related Work review vs GenNBV + VIN-NBV (2026-01-07)

## Task summary
- Review our paper’s abstract (`docs/typst/paper/main.typ`) plus
  `docs/typst/paper/sections/01-introduction.typ` and
  `docs/typst/paper/sections/02-related-work.typ`.
- Cross-check claims against the GenNBV and VIN-NBV papers and our local
  literature reviews.
- Add/adjust citations so any GenNBV/VIN-NBV-derived claims in these sections
  are explicitly cited.

## Sources reviewed
**Our paper**
- `docs/typst/paper/main.typ` (abstract)
- `docs/typst/paper/sections/01-introduction.typ`
- `docs/typst/paper/sections/02-related-work.typ`

**Local literature reviews**
- `docs/contents/literature/gen_nbv.qmd`
- `docs/contents/literature/vin_nbv.qmd`

**GenNBV (LaTeX)**
- `literature/tex-src/arXiv-GenNBV/0-Abstract.tex`
- `literature/tex-src/arXiv-GenNBV/1-Introduction.tex`
- `literature/tex-src/arXiv-GenNBV/2-Related_Work.tex`
- `literature/tex-src/arXiv-GenNBV/3-Method.tex` (reward definition)

**VIN-NBV (LaTeX)**
- `literature/tex-src/arXiv-VIN-NBV/sec/0_abstract.tex`
- `literature/tex-src/arXiv-VIN-NBV/sec/1_intro.tex`
- `literature/tex-src/arXiv-VIN-NBV/sec/2_related_work.tex`
- `literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex` (oracle RRI + CORAL/ordinal)

## Findings

### GenNBV alignment (key claims we must get right)
GenNBV’s core story is:
- RL-based NBV with **continuous 5D** free-space action (position + heading).
- **PPO** optimization.
- **Multi-source state embedding**: geometric/probabilistic occupancy from depth,
  semantics from RGB, and action history.
- **Coverage-gain reward** (coverage ratio difference) as the primary objective.

After edits, our intro/related-work statements match this and cite
`@GenNBV-chen2024` for the method details.

### VIN-NBV alignment (key claims we must get right)
VIN-NBV’s core story is:
- Coverage (and information gain) are common NBV proxies; **coverage can fail**
  under occlusions / fine details, motivating direct quality optimization.
- VIN predicts **RRI** for query views **without acquiring** that view.
- Training uses **oracle RRI** computed via explicit reconstruction + error
  reduction; they discretize RRI and train with **CORAL ordinal loss**.
- Policy is greedy: sample candidate/query views and select the best by predicted
  RRI; supports resource constraints.

After edits, our abstract/introduction/related-work align with this and cite
`@VIN-NBV-frahm2025` wherever we reuse these claims.

### Incorrect / misleading claims found
- “cluttered indoor scenes” in the abstract over-specialized VIN-NBV’s argument;
  VIN-NBV frames the issue as **complex scenes with occlusions and fine details**.
  This was corrected (wording now matches VIN-NBV’s framing).
- An earlier edit had inadvertently cited VIN-NBV for an egocentric-specific
  statement; that citation was removed to avoid misattribution.

### Background coverage / NBV history
- VIN-NBV provides a clear NBV taxonomy (coverage vs information gain; prior
  knowledge vs none; RL vs greedy ranking). We added a minimal history sentence
  to the introduction and cited VIN-NBV for it.
- If we want a more “classic NBV history” (pre-ML) beyond VIN-NBV’s brief
  discussion, we should add dedicated citations to the canonical NBV literature;
  this was kept out-of-scope per request (cite GenNBV/VIN-NBV first).

## Changes applied
- `docs/typst/paper/main.typ`: aligned the abstract with the Quarto project
  abstract (`docs/index.qmd`), then compressed it to stay concise (≤2/3 length)
  while keeping claims faithful to GenNBV/VIN-NBV and fully cited
  (`@VIN-NBV-frahm2025`, `@GenNBV-chen2024`, `@EFM3D-straub2024`,
  `@ProjectAria-ASE-2025`).
- `docs/typst/paper/sections/01-introduction.typ`: moved dropped context from the
  abstract into the introduction (object-centric VIN-NBV context; EVL
  occupancy/centerness/OBB priors) with citations.
- `docs/typst/paper/sections/01-introduction.typ`:
  - clarified GenNBV as PPO + 5-DoF + coverage-gain reward + multi-source state.
  - added VIN-NBV citations for coverage-vs-quality motivation and imitation
    learning framing.
  - cited VIN-NBV in the CORAL/ordinal-training bullet as precedent.
- `docs/typst/paper/sections/02-related-work.typ`:
  - refined the NBV paragraph to match GenNBV’s state/action/reward description.
  - cited VIN-NBV for the high-level NBV taxonomy (RL continuous vs ranking).

## Validation
- `typst compile --root docs docs/typst/paper/main.typ .codex/_render/paper.pdf` succeeds.

## Related notes
- EFM3D/EVL paragraph changes were tracked separately in
  `.codex/efm3d_related_work_review_2026-01-07.md`.

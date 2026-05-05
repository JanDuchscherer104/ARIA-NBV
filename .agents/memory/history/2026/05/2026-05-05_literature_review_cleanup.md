---
id: 2026-05-05_literature_review_cleanup
date: 2026-05-05
title: "Literature Review Cleanup"
status: done
topics: [docs, literature, glossary, thesis-scope, litkg]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/literature/index.qmd
  - docs/contents/literature/vin_nbv.qmd
  - docs/contents/literature/efm3d.qmd
  - docs/contents/literature/gen_nbv.qmd
  - docs/contents/literature/hestia.qmd
  - docs/contents/literature/project_aria.qmd
  - docs/contents/literature/pb_nbv.qmd
  - docs/contents/literature/rl_planning.qmd
  - docs/contents/literature/active_3dgs_nbv.qmd
  - docs/contents/literature/scene_script.qmd
  - docs/_quarto.yml
  - docs/typst/shared/glossary.typ
---

## Task

Implemented the literature-review cleanup plan: pruned AI-slop prose, normalized pages to a source-backed template, replaced pseudo equations/text diagrams with TeX and Mermaid, and aligned the literature hierarchy with the current target-conditioned fitted Double-Q / `Q_H` thesis path.

## Method

Used `make kg-query` with the `docs-paper-sync` profile before and after editing. Rewrote the literature index and the main paper pages around the domain hierarchy: NBV objectives/candidates, ARIA ecosystem, rollout/value/RL, 3DGS active reconstruction, and semantic scene representations.

Added minimal glossary terms for Project Aria and 3D Gaussian Splatting, regenerated glossary artifacts with `make glossary`, and moved implementation-level EVL feature-selection guidance from the EFM3D literature page into `docs/contents/impl/vin_v2_feature_proposals.qmd`.

## Outputs

- Public literature pages now use the template: core contribution, verified paper signals, ARIA-NBV adoption, do not adopt, and open risks/caveats.
- `rl_planning.qmd` now uses valid TeX for `Q_H` and Double-Q targets and Mermaid diagrams for the rollout/value ladder.
- `index.qmd` now contains the requested adoption-state table and domain hierarchy.
- Sidebar order in `docs/_quarto.yml` matches the new hierarchy.

## Verification

- `make glossary` passed and validated 51 glossary terms.
- `make kg-query KG_QUERY='source-check ARIA-NBV literature review rewrite' LITKG_PROFILE=docs-paper-sync KG_FORMAT=text` passed, with existing backend freshness warnings only.
- `rg '```text|~=|✅|⚠️|\\?\\?|You’re|Our Approach|Wikipedia|cite' docs/contents/literature` returned no hits.
- Rendered all touched literature pages individually with Quarto.
- Rendered `docs/contents/glossary.qmd`.

## Residual Notes

`make qmd-frontmatter-check` remains blocked by an existing unrelated `docs/contents/ideas.qmd` taxonomy issue (`audience: internal`, `status: archive`).

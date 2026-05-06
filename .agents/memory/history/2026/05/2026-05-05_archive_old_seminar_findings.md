---
id: 2026-05-05_archive_old_seminar_findings
date: 2026-05-05
title: "Archive Old Seminar Findings"
status: done
topics: [docs, archive, quarto]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/archive/main_seminar_findings.qmd
  - docs/contents/archive/index.qmd
  - docs/contents/experiments/findings.qmd
  - docs/_quarto.yml
  - docs/index.qmd
---

## Task

Moved stale main-seminar visual findings out of current thesis-state navigation
and into the public archive.

## Outputs

- Added `docs/contents/archive/main_seminar_findings.qmd` with archive
  frontmatter and a note that the page is historical, not current thesis
  evidence.
- Removed `docs/contents/experiments/findings.qmd`.
- Removed findings links from the homepage, navbar, and sidebar.
- Added an archive-index pointer under historical seminar material.

## Verification

- `cd docs && quarto render contents/archive/main_seminar_findings.qmd`
- `cd docs && quarto render contents/archive/index.qmd`
- `rg 'contents/experiments/findings.qmd|Experimental Findings' docs -g '!_site/**'`
- `git diff --check -- docs/_quarto.yml docs/index.qmd docs/contents/archive/index.qmd docs/contents/archive/main_seminar_findings.qmd docs/contents/experiments/findings.qmd`

The archive index render needed a temporary ignored `docs/site_libs` copy of
`bootstrap-icons.woff` from `_freeze`; the temporary directory was removed after
rendering.

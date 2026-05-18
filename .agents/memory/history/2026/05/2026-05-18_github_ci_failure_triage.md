---
id: 2026-05-18_github_ci_failure_triage
date: 2026-05-18
title: "GitHub CI Failure Triage"
status: done
topics: [ci, github-actions, docs, glossary]
confidence: high
canonical_updates_needed: []
files_touched:
  - .github/workflows/ci.yml
  - docs/typst/shared/glossary.generated.typ
---

## Task

Diagnosed the current GitHub Actions failures on `main` for commit
`98fbb36f95fa24fcc45ca22c6ec64326b8d5c877` and applied the narrow local fixes.

## Findings

The `Publish Quarto Site` workflow failed in `Check generated glossary drift`
because `scripts/glossary_build.py all` rewrote
`docs/typst/shared/glossary.generated.typ`; the committed generated file was
stale relative to the canonical glossary source.

The root `.github/workflows/ci.yml` run failed before GitHub created any job.
The run had zero check runs and no downloadable log. The workflow used
`${{ runner.temp }}` in `jobs.ci.env`, but GitHub Actions only exposes the
`runner` context in step-level environment expressions, not job-level
environment expressions.

## Outcome

Regenerated the stale Typst glossary artifact and moved `CI_RENDER_DIR` from
job-level env to the `Run root CI contract` step env in
`.github/workflows/ci.yml`.

## Verification

- `aria_nbv/.venv/bin/python scripts/glossary_build.py all`
- YAML parse check for `.github/workflows/*.yml`
- `make ci`

## Canonical State Impact

No canonical thesis or project-state update is needed. This was a CI hygiene
fix for generated docs drift and a GitHub Actions context placement error.

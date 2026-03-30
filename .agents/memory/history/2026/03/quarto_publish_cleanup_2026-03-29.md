---
id: 2026-03-29_quarto_publish_cleanup
date: 2026-03-29
title: "Move Quarto publish output out of source tree"
status: done
topics: [docs, quarto, github-pages]
confidence: high
canonical_updates_needed: []
files_touched:
  - path: docs/_quarto.yml
    kind: docs-config
  - path: .github/workflows/quarto-publish.yml
    kind: ci
  - path: docs/.gitignore
    kind: docs-config
  - path: .gitignore
    kind: repo-config
---

# Debrief

## Task
Separated Quarto source files from rendered publish output, improved docs artifact ignore rules, and added a GitHub Pages publishing workflow.

## Method
Switched Quarto output to `docs/_site`, added a Pages workflow that deploys the built site artifact, updated docs guidance, and cleaned figure organization by moving a small set of top-level assets into clearer subdirectories.

## Verification
Planned verification is `cd docs && quarto render . --no-execute` plus inspection that no source-level `*.html` files remain tracked.

## Canonical State Impact
No canonical state update was needed.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

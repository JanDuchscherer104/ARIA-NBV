---
id: 2026-05-08_aria_nbv_mermaid_skill
date: 2026-05-08
title: "ARIA-NBV Mermaid Skill"
status: done
topics: [skills, mermaid, diagrams, docs, thesis]
confidence: high
canonical_updates_needed: []
---

## Task

Added a repo-local Mermaid diagram skill and reusable tooling for ARIA-NBV
thesis diagrams.

## Method

Installed the GPT-5.5 Pro Mermaid handoff selectively: active skill routing
uses repo-style metadata, reusable references/templates/scripts/examples live
under `tools/mermaid`, and root/docs guidance points `.mmd` work to the new
skill and linter.

## Result

Mermaid figures now have a local style guide, curated Typst-to-Mermaid symbol
map, flowchart/sequence templates, an advisory/strict linter for syntax and
style issues, a render wrapper requiring global `mmdc`, and a `make
mermaid-lint` target. Existing diagrams were inventoried but not rewritten.

## Verification

Ran skill validation, Python compile checks for Mermaid scripts, shell syntax
check for the render wrapper, template linting, existing-diagram lint
inventory, `make check-agent-memory`, and `git diff --check`.

## Canonical State Impact

No thesis truth changed. The durable workflow is captured through root/docs
guidance, the new skill, and `tools/mermaid`.

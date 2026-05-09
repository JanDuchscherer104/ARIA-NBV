---
id: 2026-05-09_typst_source_linebreak_policy
date: 2026-05-09
title: Typst Source Linebreak Policy
status: done
topics:
  - typst
  - thesis
  - advisor
  - guidance
confidence: high
canonical_updates_needed: []
---

## Summary

The advisor distillation Typst source was adjusted so prose paragraphs are no longer hard-wrapped for editor display. Source line breaks in `.typ` prose should now represent intentional Typst structure such as paragraph boundaries, lists, table rows, block structure, or displayed equation layout.

## Changes

- Unwrapped prose in `docs/typst/thesis/advisor_distillation.typ` while preserving intentional Typst block, table, and equation structure.
- Added the line-break convention to the repo-local `typst-authoring` skill so future Typst edits avoid source wrapping.

## Verification

- The edited handout was compiled after the source-format pass.
- Repo memory guidance was checked with `make check-agent-memory`.

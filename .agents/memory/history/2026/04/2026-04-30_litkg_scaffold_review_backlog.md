---
id: 2026-04-30_litkg_scaffold_review_backlog
date: 2026-04-30
title: "Litkg Scaffold Review Backlog Capture"
status: done
topics: [litkg-rs, agents-db, scaffold, mcp, skills]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/.agents/issues.toml
  - .agents/external/litkg-rs/.agents/todos.toml
---

## Task

Extracted the litkg-rs scaffold review action items into the litkg-rs agents
DB under `.agents/external/litkg-rs/.agents/`.

## Method

Used the ARIA-NBV `agents-db` workflow, read the root guidance and litkg-rs
backlog schema, preserved the existing litkg-rs `ISSUE-0043` security/redaction
record, and added the review's skill, scaffold, MCP, ingestion-profile, and
context-pack work under new issue IDs.

## Outputs

- Added `ISSUE-0044` through `ISSUE-0048`.
- Added `TODO-0031` through `TODO-0046`.
- Updated `TODO-0030` to make context-pack CLI work explicit and critical.
- Added milestone-style labels to the existing litkg-rs backbone issues.

## Verification

- `make agents-db-check` passed in `.agents/external/litkg-rs`.
- `make agents-db` rendered the ranked litkg-rs issue and TODO list.

## Canonical State Impact

No ARIA-NBV canonical state files need updates. The durable change is limited to
the litkg-rs local backlog plus this debrief.

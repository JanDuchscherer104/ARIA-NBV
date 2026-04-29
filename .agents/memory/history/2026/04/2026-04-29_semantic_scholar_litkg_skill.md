---
id: 2026-04-29_semantic_scholar_litkg_skill
date: 2026-04-29
title: "Semantic Scholar litkg Skill"
status: done
topics: [skills, litkg, knowledge-graph, semantic-scholar]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - AGENTS.md
  - .gitattributes
  - .gitmodules
  - .agents/skills/semantic-scholar-litkg/SKILL.md
  - .agents/skills/semantic-scholar-litkg/references/integration-spec.md
  - .agents/memory/state/PROJECT_STATE.md
---

## Task

Create an ARIA-NBV skill for Semantic Scholar and local-source ingestion into the custom litkg-rs knowledge graph, and add litkg-rs as an agent-owned submodule.

## Method

Added `litkg-rs` under `.agents/external/litkg-rs`, initialized `semantic-scholar-litkg` with the system skill creator, and replaced the template with a concise workflow plus a detailed integration spec. The spec keeps implementation in litkg-rs, uses TOML for ARIA-NBV-specific source selection, and records graphify, Graphiti, Neo4j, code-index, MarkItDown, Context7, and mempalace integration defaults.

## Findings

The checked-out litkg-rs already has `litkg-core`, `litkg-graphify`, `litkg-neo4j`, a TOML-driven CLI, and reusable local KG scripts. `cargo info mempalace-rs` reported version `0.4.2` on 2026-04-29.

## Verification

Validation was run with the skill creator quick validator and agent-memory checks after the scaffold edits.

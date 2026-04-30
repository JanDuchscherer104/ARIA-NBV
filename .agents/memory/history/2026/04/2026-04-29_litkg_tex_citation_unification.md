---
id: 2026-04-29_litkg_tex_citation_unification
date: 2026-04-29
title: "litkg-rs TeX Citation Unification"
status: done
topics: [litkg, literature, citations, knowledge-graph]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/crates/litkg-core/src/model.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/tex.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/enrich.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/lib.rs
---

## Task

Reduce false-negative paper citation links caused by different TeX sources using different local BibTeX keys for the same paper.

## Method

Replaced the narrow citation regex with a small TeX command scanner that supports natbib and biblatex citation forms with stars and optional arguments, including `\citep[...]{...}`, `\citet*{...}`, `\parencite{...}`, and `\textcite{...}`.

Added parsed citation-reference metadata from source-local `.bib` files. Raw citation keys are preserved, while cited keys can also carry title, DOI, arXiv ID, and URL metadata. Enriched citation-edge inference now resolves local citations by exact key first, then DOI, arXiv ID, and normalized title when keys differ across source trees.

## Outputs

Reparsed and re-exported the ARIA-NBV KG. The parsed corpus still has 412 raw citations, now with 240 source-local citation-reference records across 5 parsed papers. Resolved `CITES_PAPER` edges increased to 7, with strategies: `citation_title:5`, `citation_arxiv:1`, and `exact_citation_key:1`.

## Verification

Ran `cargo fmt --all`, `cargo test`, and `make agents-db` in litkg-rs. Rebuilt ARIA parsed/materialized/Neo4j KG outputs with `./scripts/kg/ingest_papers.sh parse`, `materialize`, and `export-neo4j`.

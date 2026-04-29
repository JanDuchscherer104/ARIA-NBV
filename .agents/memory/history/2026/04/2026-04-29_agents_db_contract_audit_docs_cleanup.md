---
id: 2026-04-29_agents_db_contract_audit_docs_cleanup
date: 2026-04-29
title: "Agents DB Contract Audit And Docs Cleanup"
status: done
topics: [agents-db, docs, data-contracts, rri, scaffold]
confidence: high
canonical_updates_needed: []
files_touched:
  - scripts/agents_db.py
  - Makefile
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/refactors.toml
  - .agents/resolved.toml
  - .agents/skills/agents-db/SKILL.md
  - .agents/skills/agents-db/agents/openai.yaml
  - .agents/skills/code-review/SKILL.md
  - .agents/skills/code-review/agents/openai.yaml
  - README.md
  - docs/_quarto.yml
  - docs/index.qmd
  - docs/contents/resources.qmd
  - docs/contents/literature/index.qmd
  - docs/contents/literature/hestia.qmd
  - docs/contents/impl/coral_integration.qmd
---

Task: implement the bounded agents DB repair, M1 contract audit, and public
docs identity cleanup. The worktree was already dirty, so the change stayed
scoped to the requested agent DB, public docs, skill text, and debrief files.

Method: added `scripts/agents_db.py`, wired `make agents-db`, normalized the
TOML backlog schema, and stored detailed implementation context for the M1
data/cache/oracle gate, public docs cleanup, single-scene VIN diagnostics, docs
triage, and dirty-worktree handoff. Resolved the stale agents DB and copied
code-review skill issues through the new helper after validation passed.

Docs output: README and Quarto home now use ARIA-NBV thesis-era identity, GitHub
links point to `JanDuchscherer104/ARIA-NBV`, top-level TODO and generated agent
scaffold pages are no longer primary navigation entries, and typo-facing Quarto
pages were renamed from `coral_intergarion` to `coral_integration` and from
`hesita` to `hestia`.

Contract audit: `make context-contracts`, data-handling store/API tests, and
RRI metric tests passed. No semantic package changes were made; M1 contract
stabilization remains active in `.agents/todos.toml`.

Verification:

- `make agents-db`
- `make agents-db AGENTS_ARGS='validate'`
- `cd aria_nbv && uv run ruff format ../scripts/agents_db.py`
- `cd aria_nbv && uv run ruff check ../scripts/agents_db.py`
- `make context-contracts`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run pytest tests/rri_metrics`
- `cd docs && quarto render index.qmd && quarto render contents/roadmap.qmd && quarto render contents/questions.qmd`
- `scripts/nbv_qmd_outline.sh --compact`
- `rg -n "Seminar|hesita|coral_intergarion|JanDuchscherer104/NBV|PRML VSLAM|litkg-rs" README.md docs .agents/skills --glob '!docs/_site/**'`

Note: the single multi-file command
`cd docs && quarto render index.qmd contents/roadmap.qmd contents/questions.qmd`
still fails because this Quarto invocation passes the extra `.qmd` paths through
to Pandoc as inputs to `index.qmd`. The separate page renders above passed.

Canonical state impact: no durable project-state file changed. Active work is
tracked in `.agents/issues.toml`, `.agents/todos.toml`, `.agents/refactors.toml`,
and `.agents/resolved.toml`.

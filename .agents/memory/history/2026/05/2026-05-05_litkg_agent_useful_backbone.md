---
id: 2026-05-05_litkg_agent_useful_backbone
date: 2026-05-05
title: "litkg Agent-Useful Backbone Implementation"
status: done
topics: [litkg, kg, agent-memory, retrieval]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
files_touched:
  - .agents/external/litkg-rs/crates/litkg-core/src/backlog.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/ranking.rs
  - .agents/external/litkg-rs/crates/litkg-neo4j/src/lib.rs
  - .agents/external/litkg-rs/crates/litkg-viewer/src/query.rs
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/examples/benchmarks/local_repo_qa/aria-nbv.jsonl
  - .agents/external/litkg-rs/examples/benchmarks/local_repo_qa/catalog.toml
  - .agents/external/litkg-rs/scripts/benchmarks/repo_qa_lib.py
  - Makefile
---

# litkg Agent-Useful Backbone Implementation

## Task
Implemented the accepted plan to make litkg-rs more useful as ARIA-NBV's agent retrieval backbone, with emphasis on trust, freshness, backlog coverage, and source typing.

## Outputs
- Added a reusable litkg-core backlog loader for active issues and todos.
- Exported active agents-db records as `AgentBacklogIssue` and `AgentBacklogTodo` nodes with reference edges.
- Made `kg find` refresh stale Neo4j export bundles before searching.
- Added source type, authority, freshness, final score, and ranking rationale to graph search hits.
- Fixed authority-tier matching for absolute paths by allowing relative suffix matches.
- Split non-literature parsed documents into document/research-note/transcript graph labels instead of treating every parsed source as literature.
- Added `kg consolidate` and the root `make kg-consolidate` wrapper as proposal-only consolidation output.
- Added claim-check overclaim risk detection for finished/deployed/end-to-end RL-policy claims.
- Added an ARIA-NBV local RepoQA benchmark fixture covering RRI, VIN, rollout, cache/offline-store, LRZ, docs, and KG workflow routing.
- Extended the deterministic RepoQA harness with hidden-file `file_contains` path questions so `.agents` memory and skills can be exact-match retrieval targets.

## Verification
- `cargo fmt --all`
- `cargo test -p litkg-core -p litkg-neo4j -p litkg-viewer -p litkg-cli`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- kg find --config .configs/litkg.toml --repo-root . --limit 8 --format json --modality backlog litkg`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- kg find --config .configs/litkg.toml --repo-root . --limit 5 --format json --no-rg "current thesis spine"`
- `make kg-claim-check KG_CLAIM="ARIA-NBV is a finished end-to-end RL policy" KG_FORMAT=json`
- `make kg-consolidate KG_FORMAT=json`
- `python3 scripts/benchmarks/run_repo_qa.py --dataset examples/benchmarks/local_repo_qa/aria-nbv.jsonl --repo-root /home/jd/repos/ARIA-NBV --answerer direct --output-path /tmp/aria-nbv-repoqa-result.json --artifact-dir /tmp/aria-nbv-repoqa-artifacts --persist-artifact-dir examples/benchmarks/local_repo_qa/artifacts/aria-nbv`
- `cargo run -p litkg-cli -- benchmark validate --catalog examples/benchmarks/local_repo_qa/catalog.toml --results examples/benchmarks/local_repo_qa/latest-results.toml`
- `cargo run -p litkg-cli -- benchmark run --catalog examples/benchmarks/local_repo_qa/catalog.toml --integrations examples/benchmarks/local_repo_qa/integrations.toml --plan examples/benchmarks/local_repo_qa/run-plan.toml --benchmark-id aria-nbv-local-repo-qa --output /tmp/aria-nbv-local-repo-qa-results.toml`
- `make check-agent-memory`
- `make agents-db` inside `.agents/external/litkg-rs`

## Canonical State Impact
The implementation changes litkg from an operator-oriented KG surface toward an agent-memory retrieval plane. Canonical memory should be updated after reviewing whether this behavior is now the durable expected contract for broad ARIA-NBV agent work.

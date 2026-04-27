---
name: code-review-litkg-rs
description: Use when reviewing litkg-rs changes in the working tree or on a GitHub pull request, especially to produce severity-ranked findings with file and line references and to gate autoresearch winner branches before promotion.
---

# Code Review For litkg-rs

## When To Use

Use this skill when the task is to:

- review the current working tree before commit or branch promotion
- review a GitHub pull request diff or requested-review state
- run a review gate inside an autoresearch loop before a trial is kept
- aggregate accepted review feedback into follow-up edits or backlog items

Do not use this skill for:

- pure implementation without a review ask
- general architecture brainstorming without concrete diffs
- replying to GitHub review threads without also using `github:gh-address-comments` when thread state matters

## Grounding

Before reviewing substantial changes, read:

1. `AGENTS.md`
2. `CODEOWNER.md`
3. `README.md`
4. `docs/architecture.md`
5. `.agents/AGENTS_INTERNAL_DB.md`

When the review is benchmark-driven or autoresearch-driven, also read:

- `docs/benchmarks.md`
- `.agents/skills/autoresearch-litkg-rs/SKILL.md`
- the active `.logs/autoresearch/<tag>/brief.md` when it exists

## Review Standard

Default to a code-review mindset:

- findings first
- order by severity
- focus on correctness, behavioral regressions, determinism drift, missing validation, repo-boundary leaks, and operator-facing contract breaks
- include tight file and line references whenever the location is clear
- if there are no findings, say that explicitly and call out residual risk or missing tests

Use this severity rubric:

- `P0`: data loss, security break, unrecoverable crash, or merge-blocking build or test failure
- `P1`: correctness bug, regression, boundary violation, or broken contract likely to matter immediately
- `P2`: maintainability, observability, or test gap that should be fixed soon
- `P3`: minor polish or documentation issue

## Working Tree Review

1. Establish the review surface with:

```bash
git status --short
git diff --stat
git diff
```

2. If the tree is large, narrow by file or subsystem before drawing conclusions.
3. Review tests and docs alongside code changes when contracts moved.
4. Run the narrowest validating commands that can confirm or falsify likely findings.
5. Report:
   - findings
   - open questions or assumptions
   - brief change summary only after findings

Treat these as first-class review targets in this repo:

- deterministic output guarantees
- adapter boundary leaks between core, graphify, and Neo4j
- repo-specific assumptions accidentally entering `litkg-core`
- benchmark schema or autoresearch-target contract drift
- Apple Silicon-hostile local tooling choices when they affect operator UX

## Pull Request Review

1. Resolve the PR from a URL, `<owner>/<repo>#<number>`, or the current branch PR.
2. Gather diff and metadata with the lightest tool that works:
   - local git when the branch and base are both present
   - GitHub app, `gh pr view`, or `gh pr diff` when PR context is needed
3. When the task depends on unresolved review threads, inline anchors, or resolution state, use `github:gh-address-comments` instead of guessing from flat comments.
4. Separate:
   - new independent findings from your review
   - existing requested changes already on the PR
   - informational comments that do not need a code change
5. Do not submit a review, reply on GitHub, or resolve threads unless the user explicitly asks.

Useful PR commands:

```bash
gh pr view --json number,title,url,baseRefName,headRefName,reviewDecision
gh pr diff
```

## Autoresearch Review Gate

Apply this skill to every candidate winning experiment before promotion.

1. Run the frozen evaluation harness first.
2. Review the trial diff against the current winner branch or merge base, not against an unrelated dirty tree.
3. Block promotion when there are unresolved `P0` or `P1` findings.
4. If only `P2` or `P3` findings remain, either:
   - fix them in the same trial before promotion, or
   - keep the winner and record the debt explicitly in the run log or backlog
5. If the experiment changes benchmark schema, rendered targets, docs, or operator workflow, review those artifacts as product surfaces, not just the Rust diff.
6. If the experiment leaks outside the declared mutable surface, treat that as at least `P1` unless the brief explicitly changed.

For benchmark-driven runs, verify that:

- `make benchmark-validate` still matches the reviewed artifact shape
- `make autoresearch-target AUTORESEARCH_TARGET_ID=<target>` still renders the intended operator prompt
- any accepted contract change is reflected in `README.md`, `docs/benchmarks.md`, and `.agents/AGENTS_INTERNAL_DB.md`

## Fan-Out

If the user explicitly asks for delegated or parallel review:

- split review ownership by file area or concern, with disjoint scopes
- keep the aggregation pass local
- do not ask sub-agents to submit reviews or resolve GitHub threads directly

## Output Shape

Use:

1. `Findings`
2. `Open Questions / Assumptions`
3. `Change Summary` or `Residual Risk`

When there are no findings, say:

- no findings identified
- what you did validate
- what remains unvalidated

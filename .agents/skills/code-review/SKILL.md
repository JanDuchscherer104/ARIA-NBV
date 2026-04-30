---
name: code-review-aria-nbv
description: Use when reviewing ARIA-NBV changes in the working tree or on a GitHub pull request, especially to produce severity-ranked findings with file and line references for Python, Quarto, Typst, config, and agent-memory changes.
---

# Code Review For ARIA-NBV

## When To Use

Use this skill when the task is to:

- review the current working tree before commit or branch promotion
- review a GitHub pull request diff or requested-review state
- review agent-memory, docs, data-contract, RRI, VIN, or Streamlit changes
- aggregate accepted review feedback into follow-up edits or agents DB records

Do not use this skill for:

- pure implementation without a review ask
- general architecture brainstorming without concrete diffs
- replying to GitHub review threads without also using `github:gh-address-comments` when thread state matters

## Grounding

Before reviewing substantial changes, read:

1. `AGENTS.md`
2. the nearest nested `AGENTS.md` for the touched surface
3. `README.md`
4. `.agents/memory/state/PROJECT_STATE.md`
5. `.agents/AGENTS_INTERNAL_DB.md`

For docs-heavy reviews, also read `docs/AGENTS.md` and start from
`docs/typst/seminar_paper/main.typ`. For package-contract reviews, use
`make context-contracts` before broad source browsing.

## Review Standard

Default to a code-review mindset:

- findings first
- order by severity
- focus on correctness, behavioral regressions, determinism drift, missing validation, repo-boundary leaks, frame/pose mistakes, stale documentation, and operator-facing contract breaks
- include tight file and line references whenever the location is clear
- if there are no findings, say that explicitly and call out residual risk or missing tests
- Prefer tests that exercise public contracts and user-visible behavior. Flag
  tests that only preserve private helper shape when behavior could regress.
- For risky behavior changes, prefer tracer-bullet evidence: one failing
  public-interface behavior test, minimal implementation, repeat.

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
5. Report findings, then open questions or assumptions, then a brief change summary.

Treat these as first-class review targets in this repo:

- `PoseTW` / `CameraTW` frame consistency and display-only CW90 corrections
- config-as-factory usage through `.setup_target()`
- immutable VIN offline-store and split semantics
- RRI metric meaning, binning semantics, and logged metric names
- VIN candidate-ranking contracts and validation defaults
- docs alignment with `docs/typst/seminar_paper/main.typ`
- agent-memory/debrief hygiene under `.agents/memory/`

## Pull Request Review

1. Resolve the PR from a URL, `<owner>/<repo>#<number>`, or the current branch PR.
2. Gather diff and metadata with the lightest tool that works:
   - local git when the branch and base are both present
   - GitHub app, `gh pr view`, or `gh pr diff` when PR context is needed
3. When the task depends on unresolved review threads, inline anchors, or resolution state, use `github:gh-address-comments` instead of guessing from flat comments.
4. Separate new independent findings from existing requested changes and informational comments.
5. Do not submit a review, reply on GitHub, or resolve threads unless the user explicitly asks.

Useful PR commands:

```bash
gh pr view --json number,title,url,baseRefName,headRefName,reviewDecision
gh pr diff
```

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
- what you validated
- what remains unvalidated

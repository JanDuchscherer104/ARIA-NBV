---
name: code-review-aria-nbv
description: Use when reviewing Aria-NBV working tree changes or a GitHub pull request, especially to produce severity-ranked findings with file and line references across data handling, rendering, pose generation, VIN, Lightning, RL, app, docs, or agent scaffold surfaces.
---

# Code Review For Aria-NBV

## When To Use

Use this skill when the task is to:

- review the current working tree before commit or branch promotion
- review a GitHub pull request diff or requested-review state
- review a scoped subsystem change such as data handling, rendering, pose generation, VIN, Lightning, RL, Streamlit, docs, or scaffold work
- aggregate accepted review feedback into follow-up edits or backlog items

Do not use this skill for:

- pure implementation without a review ask
- broad architecture brainstorming without concrete diffs
- replying to GitHub review threads when thread-level state requires the GitHub comment workflow skill instead

## Grounding

Before reviewing substantial changes, read:

1. `AGENTS.md`
2. `README.md`
3. `.agents/AGENTS_INTERNAL_DB.md`
4. the nearest nested `AGENTS.md`
5. the relevant canonical memory doc when the review touches current project truth

For docs or thesis-surface reviews, also read:

- `docs/typst/paper/main.typ`
- `.agents/memory/state/PROJECT_STATE.md`
- `.agents/memory/state/OPEN_QUESTIONS.md` when the diff changes stated priorities or open research questions

## Review Standard

Default to a code-review mindset:

- findings first
- order by severity
- focus on correctness, contract drift, behavioral regressions, frame/shape mistakes, missing validation, stale docs, and missing tests
- include tight file and line references when the location is clear
- if there are no findings, say that explicitly and call out residual risk or missing validation

Use this severity rubric:

- `P0`: destructive data loss, unrecoverable crash, or merge-blocking build/test failure
- `P1`: correctness bug, contract break, or likely regression
- `P2`: maintainability, observability, or test/documentation gap that should be fixed soon
- `P3`: minor polish issue

## Working Tree Review

1. Establish the review surface with:

```bash
git status --short
git diff --stat
git diff
```

2. If the tree is large, narrow by file area before drawing conclusions.
3. Review tests and docs alongside code changes when contracts moved.
4. Run the narrowest validating commands that can confirm likely findings.
5. Report:
   - findings
   - open questions or assumptions
   - brief change summary only after findings

Treat these as first-class review targets in this repo:

- `PoseTW` / `CameraTW` contract drift
- candidate-frame or display-only frame corrections leaking into training/runtime semantics
- typed container or config-as-factory regressions
- stale Quarto/Typst/paper alignment
- RL or counterfactual planning claims that exceed the implemented evidence
- scaffold guidance, hook, or validator drift

## Pull Request Review

1. Resolve the PR from a URL, `<owner>/<repo>#<number>`, or the current branch PR.
2. Gather diff and metadata with the lightest tool that works:
   - local git when branch and base are present
   - GitHub CLI or app tools when PR context is needed
3. When unresolved review threads or inline comment state matter, use the GitHub review-comment workflow rather than guessing from flat comments.
4. Separate:
   - new independent findings from your review
   - existing requested changes already on the PR
   - informational comments that do not need code changes
5. Do not submit a GitHub review or resolve threads unless explicitly asked.

---
name: create-gh-pr
description: Create, update, or publish GitHub pull requests for Aria-NBV. Use when Codex needs to write a PR title/body for this repo, update an existing PR description, or publish a branch with reviewer-friendly work packages and explicit validation notes.
---

# Create GitHub PR For Aria-NBV

## Overview

Create PRs that explain the branch as a small set of reviewer-friendly work packages instead of a raw diff dump.

## Workflow

1. Ground the branch.
   - Read `AGENTS.md`, `README.md`, `.agents/AGENTS_INTERNAL_DB.md`, and the nearest nested `AGENTS.md` for touched subtrees.
   - Inspect `git status -sb`, `git diff --stat`, and key diffs against the intended base branch.
   - Separate unrelated local edits before staging or publishing.
2. Define the PR shape.
   - Group the diff into 2-7 work packages.
   - Name packages by repo boundary or contract, not by commit order.
   - Prefer scopes such as `counterfactual rollout core`, `rl environment`, `streamlit rl inspector`, `docs and memory sync`, `mojo benchmark reporting`, or `agent scaffolding`.
3. Validate before writing.
   - Use the narrowest real commands that match the touched surface.
   - Typical commands here are `make check-agent-scaffold`, `make check-agent-memory`, `make context`, targeted `uv run pytest`, and targeted Ruff checks.
   - If a command could not be run, say so explicitly in the PR body.
4. Write the title.
   - Use a concise sentence that describes the branch outcome.
   - Prefer repo language such as `counterfactual`, `RRI`, `VIN`, `rendering`, `Lightning`, `RL`, `docs`, or `agent scaffold`.
5. Create or update the PR with explicit `gh` commands.

## CLI Path

```bash
gh pr create \
  --draft \
  --base "$BASE" \
  --head "$(git branch --show-current)" \
  --title "$TITLE" \
  --body-file "$BODY_FILE"
```

```bash
gh pr edit <number-or-branch> \
  --title "$TITLE" \
  --body-file "$BODY_FILE"
```

## PR Body Shape

```md
## Summary
<one paragraph describing the branch and why it matters>

Focused verification completed:
- `make check-agent-scaffold`
- `uv run pytest ...`

## Work Packages
| WP | Scope | Primary surfaces | Status |
| --- | --- | --- | --- |
| WP1 | Counterfactual rollout core | `aria_nbv/aria_nbv/pose_generation` | resolved |
```

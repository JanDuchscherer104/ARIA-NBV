---
name: gh-issue-lifecycle
description: Use when creating, syncing, triaging, or resolving GitHub issues for Aria-NBV. Covers issue discovery, de-duplication, issue-body structure, local `.agents` DB sync, and issue-closing workflow through linked PRs.
---

# GitHub Issue Lifecycle For Aria-NBV

## Overview

Use this skill when a user asks to create GitHub issues, mirror the local `.agents` backlog into GitHub, improve issue quality, or resolve an issue cleanly from issue body through merged change.

For this repo, the source-of-truth surfaces are:

- `.agents/issues.toml`
- `.agents/todos.toml`
- `.agents/resolved.toml`
- `.github/ISSUE_TEMPLATE/01-backlog-item.yml`
- `.github/pull_request_template.md`

## Create Issues

1. Read the local `.agents` DB and list existing GitHub issues before creating anything.
2. De-duplicate by local ID and title. If synced, the GitHub title should usually start with `[ISSUE-xxxx]`.
3. Use the repo issue-template shape even when creating via CLI or API.
4. The body should usually contain:
   - `Target`
   - `Current problem`
   - `Required change`
   - `Acceptance criteria`
   - `Test expectations`
   - `Related local IDs`
   - optional `Expected touch set`
5. Prefer concrete current-repo evidence over abstract feature intent.

## Resolve Issues

1. Start from the GitHub issue and linked local DB entry.
2. Restate the acceptance criteria in your own words before coding.
3. Implement the change and run the relevant repo validation.
4. Update the local DB and move completed records into `.agents/resolved.toml`.
5. When opening or updating a PR, include `Closes #<number>` when appropriate.

---
description: Run make kg-claim-check on an advisor-facing claim.
allowed-tools: Bash(make kg-claim-check:*), Bash(make kg-capabilities:*)
argument-hint: "<claim text>"
---

Run `make kg-claim-check KG_CLAIM="$ARGUMENTS" KG_FORMAT=json` and summarize:

- supporting evidence with source role and authority/freshness;
- conflicting or missing evidence;
- whether the claim is supported, partially supported, or unsupported.

If `make kg-capabilities KG_FORMAT=json` shows the KG is degraded, fall back
to inspecting cited canonical sources directly per
`.agents/references/source_order.md` and report that fallback explicitly.

Required for advisor-facing proposal, roadmap, research-question, or
literature-synthesis claims before they are treated as supported.

# Ruthless Simplification

Use only when the user explicitly asks for ruthless, aggressive, or drastic
reduction.

## Preserve

- behavior exercised by tests
- explicit external or public contracts listed in `AGENTS.md`
- domain invariants
- intentionally retained CLI, API, or config behavior named in the task

## Do Not Preserve Unless Asked

- backward compatibility shims
- deprecated wrappers and aliases
- internal symbol names
- transitional overloads
- dead feature flags
- duplicate DTO hierarchies or enums
- legacy extension points with no active consumers

## Bias

- Prefer deletion over abstraction.
- Prefer inlining over helper extraction.
- Prefer one canonical type per concept.
- Prefer one canonical enum per semantic axis.
- If a new abstraction is introduced, it must protect a real invariant, serve
  meaningful reuse, become the canonical owner, or remove more code than it
  adds.

## Report

Before editing, state what will be deleted, merged, inlined, or removed as a
compatibility path. After editing, report files/symbols removed, LOC delta,
tests run, and remaining risks.

---
id: 2026-05-11_data_handling_rollout_arch_readme
date: 2026-05-11
title: "Data Handling Rollout Architecture README"
status: done
topics: [data-handling, rollout-store, sharding, documentation]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/README.md
---

## Task

Outlined the target data-handling and rollout-store architecture in the package
README, with emphasis on clean physical separation, low redundancy, sharded
multi-step rollout stores, and a joined reader API.

## Method

Read the data-handling owner guidance, current package exports, rollout dataset
writer, rollout Zarr store, rollout invalidity contract, and canonical decision
memory. Updated the README only for the public architecture surface.

## Output

The README now distinguishes the immutable `vin_offline` one-step store from
the target sharded `rollouts_v1` sidecar. It includes ARIA-styled Mermaid flow
and class/UML diagrams, a rollout collection tree, an individual multi-step
sample tree with symbols and shapes, a selected-action depth-retention
contract, a multi-step oracle generation sequence diagram, and verification
commands.

## Verification

Ran `git diff --check -- aria_nbv/aria_nbv/data_handling/README.md`.
Extracted the Mermaid fences to `/tmp` and ran
`python3 tools/mermaid/scripts/aria_mermaid_lint.py`; the linter reported no
errors and no warnings across the four README Mermaid blocks. Local render
validation was skipped because `mmdc` was not installed in the active
environment.

## Canonical State Impact

No separate canonical state update is required. The README reflects existing
decisions that rollout replay belongs in standalone rollout artifacts and that
full meshes/backbone tensors remain referenced rather than duplicated.

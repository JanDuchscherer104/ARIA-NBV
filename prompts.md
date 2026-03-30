---
spec_kind: autoimprove_gt
name: nbv-autoimprove
version: 1
repo_root: .
package_root: aria_nbv/aria_nbv
tests_root: aria_nbv/tests
default_mode: audit
editable_paths:
  - aria_nbv/aria_nbv
  - aria_nbv/tests
  - prompts.md
protected_paths:
  - .agents
  - docs/_freeze
  - docs/_site
  - external
mirror_roots:
  - left: aria_nbv/aria_nbv/data
    right: aria_nbv/aria_nbv/data_handling
  - left: aria_nbv/aria_nbv/vin
    right: aria_nbv/aria_nbv/vin/experimental
shared_helper_roots:
  - aria_nbv/aria_nbv/utils
  - aria_nbv/aria_nbv/data_handling
ignore_duplicate_function_names:
  - main
  - summarize
ignore_duplicate_class_names:
  - PlottingConfig
modes:
  audit:
    summary: Map duplicate types, models, helpers, and feature-contact points before editing.
    focus:
      - Localize one-source-of-truth violations across adjacent package roots.
      - Trace where legacy compatibility layers are still consumed.
      - Verify which surfaces are active, deprecated, experimental, or dead.
    required_outputs:
      - Ranked redundancy map with exact file paths.
      - Feature inventory for the touched surface.
      - Proposed deletion and migration order.
    verification:
      - No edits required.
      - Validate claims against code search, tests, and docs.
  dedupe:
    summary: Collapse adjacent duplicate definitions into one canonical implementation.
    focus:
      - Replace mirrored modules with imports or re-exports where migration still needs compatibility.
      - Move shared helpers into shared utility modules instead of leaf files.
      - Delete obsolete interfaces instead of adding new shims.
    required_outputs:
      - One canonical definition per shared type, model, and helper.
      - Reduced line count and reduced public surface area.
      - Updated tests for the kept implementation.
    verification:
      - ruff format
      - ruff check
      - targeted pytest for the touched surface
  libraryize:
    summary: Prefer stable external libraries when they remove local boilerplate without obscuring behavior.
    focus:
      - Replace custom serialization or config glue with existing deps when the repo already depends on them.
      - Remove thin wrappers that only rename third-party functionality.
      - Keep call sites typed and explicit after replacement.
    required_outputs:
      - Fewer local helper lines.
      - Clearer dependency boundary.
      - No regression in targeted behavior.
    verification:
      - ruff format
      - ruff check
      - targeted pytest for the touched surface
  verify:
    summary: Lock behavior down with focused tests before or after simplification.
    focus:
      - Capture feature parity for code being deduplicated.
      - Add regression tests around public contracts and compatibility imports.
      - Prefer small, direct tests over broad integration scaffolding.
    required_outputs:
      - Focused tests for the canonical surface.
      - Clear failure mode when an old path is intentionally retired.
      - Updated verification notes for the next round.
    verification:
      - ruff check
      - targeted pytest for the touched surface
cost_function:
  target: minimize
  hard_gates:
    - Preserve behavior on the touched surface.
    - Run format, lint, and targeted pytest for touched modules.
    - New helper logic must live in shared helper modules, not UI panels or model leaf files.
    - Prefer one canonical definition per adjacent type, model, or helper.
    - Compatibility re-exports are allowed only when there is still a live caller or test that needs them.
  weights:
    python_loc: 0.02
    mirrored_module_pairs: 40.0
    exact_duplicate_class_defs: 20.0
    exact_duplicate_function_defs: 12.0
    repeated_public_class_names: 10.0
    repeated_private_helper_names: 8.0
prompt_sections:
  objective: >-
    Simplify, declutter, and reduce total lines in aria_nbv while preserving
    behavior. Prefer external libraries over local boilerplate, collapse
    duplicate adjacent definitions into one source of truth, and improve test
    coverage as redundancy is removed.
  repo_hotspots: >-
    Highest priority is the data versus data_handling split, followed by legacy
    or experimental VIN surfaces that still duplicate helpers, types, or model
    plumbing. Helper logic should converge into shared utility modules.
  loop: >-
    Each turn should explicitly choose one mode: audit, dedupe, libraryize, or
    verify. Use audit when the next deletion seam is unclear, dedupe when a
    canonical source is already obvious, libraryize when a dependency can
    replace local code, and verify whenever simplification changes a public
    contract or compatibility edge.
notes:
  - Keep a single VIN offline dataset direction and avoid parallel cache stacks.
  - Cross-check all contact points between data, data_handling, Lightning, VIN, and diagnostics before deleting compatibility paths.
  - Preserve the option to add K counterfactual trajectories per snippet without multiplying dataset surfaces.
  - Simplify metadata handling instead of spreading metadata logic across multiple cache layers.
---

# NBV Autoimprove Ground Truth

This file is the single canonical prompt and cost-function spec for running
Karpathy-style autoimprove on this repository.

## Mission

The objective is not to add more framework. The objective is to delete
redundancy, reduce moving parts, keep features working, and leave behind the
smallest defensible implementation. The highest-value wins are:

- collapsing adjacent duplicate type and model definitions into one source of truth,
- moving repeated helpers into shared utility modules,
- replacing local boilerplate with external libraries that are already stable dependencies,
- and increasing confidence through focused tests on the kept surface.

## Repo-Specific Priorities

The immediate hotspot is the `aria_nbv.data` versus `aria_nbv.data_handling`
split. Treat that seam as the default audit target unless another adjacent
duplication seam has higher impact. The current long-term direction is:

- maintain one active offline VIN dataset direction,
- simplify metadata ownership and serialization,
- and keep the codebase ready for optional K-counterfactual trajectories
  without multiplying cache and dataset APIs.

## Mode Playbook

Start in `audit` whenever the next deletion seam is ambiguous. Switch to
`dedupe` once the canonical implementation is obvious. Use `libraryize` when a
third-party dependency can replace custom code. Use `verify` before or after a
risky simplification when behavior needs to be pinned down.

Every turn should report:

1. the selected mode,
2. the specific seam being simplified,
3. the feature checks used to guard the change,
4. and the next best follow-up seam if the current one lands cleanly.

## Existing Owner Guidance

The user wants a cross-check of everything implemented in `data`,
`data_handling`, and all contact points across `aria_nbv`. This includes:

- what classes and interfaces are actually needed,
- what is currently implemented and actively used,
- what redundancy can be removed for maximum reduction in moving parts,
- how metadata handling can be simplified,
- and how to avoid parallel dataset surfaces for the same underlying VIN data.

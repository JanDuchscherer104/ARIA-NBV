---
name: nbv-autoimprove
version: 1
executor: codex
report_dir: .agents/workspace/autoimprove/reports
diff_base: origin/main
audit_paths:
  - aria_nbv/aria_nbv
editable_paths:
  - aria_nbv/aria_nbv
  - aria_nbv/tests
protected_paths:
  - external
  - docs/_generated
  - docs/_freeze
  - docs/_site
  - .agents/memory/state
adjacent_module_groups:
  - name: data_to_data_handling
    left: aria_nbv/aria_nbv/data
    right: aria_nbv/aria_nbv/data_handling
    min_similarity: 0.75
    canonical_owner: aria_nbv/aria_nbv/data_handling
  - name: vin_to_vin_experimental
    left: aria_nbv/aria_nbv/vin
    right: aria_nbv/aria_nbv/vin/experimental
    min_similarity: 0.55
    canonical_owner: aria_nbv/aria_nbv/vin
  - name: lightning_old_to_current
    left: aria_nbv/aria_nbv/lightning/lit_module_old.py
    right: aria_nbv/aria_nbv/lightning
    min_similarity: 0.55
    canonical_owner: aria_nbv/aria_nbv/lightning
canonical_owner_rules:
  - name: prefer_data_handling_over_data
    matches:
      - aria_nbv/aria_nbv/data
      - aria_nbv/aria_nbv/data_handling
    prefer: aria_nbv/aria_nbv/data_handling
  - name: prefer_main_vin_over_experimental
    matches:
      - aria_nbv/aria_nbv/vin
      - aria_nbv/aria_nbv/vin/experimental
    prefer: aria_nbv/aria_nbv/vin
  - name: prefer_current_lightning_over_old
    matches:
      - aria_nbv/aria_nbv/lightning
      - aria_nbv/aria_nbv/lightning/lit_module_old.py
    prefer: aria_nbv/aria_nbv/lightning
import_preference_rules:
  - name: prefer_data_handling_imports
    legacy_prefix: aria_nbv.data
    canonical_prefix: aria_nbv.data_handling
    allow_paths:
      - aria_nbv/aria_nbv/data
shared_helper_roots:
  - aria_nbv/aria_nbv/utils
ignored_helper_names:
  - main
  - target
  - setup_target
  - forward
  - to
  - load
  - get
  - run
  - __getattr__
helper_density_threshold: 6
cost_function:
  objective: maximize
  expression: "50*verification_pass_rate + 2.0*coverage_delta + 0.01*net_python_lines_removed - 16*duplicate_module_pairs - 10*repeated_class_groups - 9*helper_collisions - 1.5*private_helper_functions - 6*private_export_names - 8*legacy_import_edges - 25*protected_path_touches"
  weights:
    verification_pass_rate: 50.0
    coverage_delta: 2.0
    net_python_lines_removed: 0.01
    duplicate_module_pairs: -16.0
    repeated_class_groups: -10.0
    helper_collisions: -9.0
    private_helper_functions: -1.5
    private_export_names: -6.0
    legacy_import_edges: -8.0
    protected_path_touches: -25.0
modes:
  simplify:
    goal: Collapse duplicate implementations into a single owner, prefer wrapper or re-export compatibility layers over copied definitions, and reduce Python LOC without changing behavior.
    focus:
      - data/data_handling duplication
      - adjacent type and model definitions
      - duplicated cache and config surfaces
      - helpers that belong in shared modules
    verify_commands:
      - cd aria_nbv && uv run pytest tests/data tests/data_handling tests/lightning/test_vin_datamodule_sources.py tests/lightning/test_vin_batch_collate.py
  verify:
    goal: Prove behavior parity after simplification and catch compatibility regressions early.
    focus:
      - compatibility imports
      - dataset and cache parity
      - public package contracts
    verify_commands:
      - cd aria_nbv && uv run pytest tests/data tests/data_handling tests/lightning tests/vin
  coverage:
    goal: Add narrow regression tests around newly simplified seams so deletion becomes safe.
    focus:
      - compatibility wrappers
      - cache readers and writers
      - VIN batch and snippet adapters
    verify_commands:
      - cd aria_nbv && uv run pytest tests/data tests/data_handling tests/utils
  externalize:
    goal: Replace custom code with existing library or existing repo helper code whenever that shortens the implementation and preserves behavior.
    focus:
      - serialization helpers
      - index and json handling
      - config plumbing
    verify_commands:
      - cd aria_nbv && uv run pytest tests/data_handling tests/utils
  feedback:
    goal: Produce the next simplification target list from the latest audit output.
    focus:
      - highest-overlap module pairs
      - highest-churn compatibility surfaces
      - legacy imports that bypass canonical owners
      - public API leakage through underscore exports
      - missing tests blocking deletion
    verify_commands: []
default_mode: simplify
---

# NBV Autoimprove

This file is the single ground-truth specification for autonomous code
improvement in the Aria-NBV Python project. It plays the role that
`program.md` plays in `karpathy/autoresearch`, except the loop here is
bounded, seam-based, and code-reviewable instead of unconstrained and
free-running.

## Scope

Work across the full `aria_nbv` package and its tests, not just the currently
open PR. The immediate target is the Python codebase, especially the duplicated
or adjacent surfaces that currently hide the real architecture behind copied
types, copied configs, copied helper functions, and compatibility layers that
grew into independent implementations.

## Operating Rules

1. One owner per concept. If two adjacent modules define the same type, config,
   cache contract, or helper, keep a single canonical implementation and make
   the legacy path a thin compatibility surface only when compatibility is still
   needed.
2. Prefer deletion to rearrangement. A smaller codebase with unchanged behavior
   is a better outcome than a large refactor that mostly moves code around.
3. Helpers belong in shared helper roots, not in feature modules that happen to
   need them first.
4. Use an existing external library when it removes bespoke code and does not
   make the public contract harder to understand.
5. Every simplification pass must leave behind a verification seam: targeted
   tests, contract checks, or a clearly named compatibility wrapper.

## Code-Specific Adaptation Of Karpathy Autoresearch

The transferable idea is not "let the agent rewrite everything forever". The
transferable idea is:

- keep a single machine-readable spec for the objective,
- let the agent switch modes on each turn,
- make the score depend on measurable code quality signals,
- keep the editable scope explicit,
- and treat unchanged behavior with less code as a first-class win.

For Aria-NBV, the editable unit is not one file like `train.py`. It is one
localized simplification seam at a time, chosen from the audit output. The
executor is Codex itself: it reads this file, picks the requested mode, edits
code within `editable_paths`, and runs the configured verification commands.

## Setup

1. Read this file and render the active mode prompt.
2. Run the audit and inspect duplicate modules, repeated contracts, helper
   collisions, and private helper sprawl.
3. Pick one simplification seam with a clear canonical owner and a focused
   verification path.
4. Edit only the minimum code needed for that seam.

## Iteration Loop

1. Run the audit for the active mode.
2. Pick one duplication seam with a clear owner.
3. Simplify by deleting duplicate definitions or turning old paths into thin
   wrappers over the canonical implementation.
4. Run the mode's verification commands.
5. Score the result and emit a report.
6. Record what is still duplicated and what test gap still blocks deletion.
7. When the owner explicitly requests autonomous iteration, immediately select
   the next bounded seam after each verified pass instead of waiting for more
   feedback, and stop only on a real blocker.

## Code-Specific Rules

- Prefer one canonical owner per contract, config, cache record, batch/view
  type, and dataset surface.
- Treat `aria_nbv.data` as a compatibility surface when it duplicates
  `aria_nbv.data_handling`.
- Treat `aria_nbv.vin.experimental` and `lightning/lit_module_old.py` as
  historical or compatibility surfaces unless the current active runtime still
  depends on them.
- Private top-level helpers outside `aria_nbv/aria_nbv/utils` are a liability
  by default. Inline them into the owner or move them into a shared helper
  module when they genuinely serve multiple modules.
- Prefer scoring changes that remove repeated contracts and helper collisions
  over cosmetic net-LOC wins. Shorter code only counts as a win when ownership
  and verification get simpler too.

## Preferred Simplification Moves

- Collapse `aria_nbv.data` duplicates into `aria_nbv.data_handling` ownership.
- Turn legacy modules into thin compatibility wrappers instead of maintaining
  copied class and function bodies.
- Move shared helper logic into `aria_nbv.utils` or a clear surface-local
  helper module.
- Prefer small external libraries over bespoke infrastructure when they remove
  code without obscuring the public contract.
- Delete obsolete or experimental code paths when they are no longer part of
  the active architecture.

## Stop Conditions

Stop a concrete simplification pass when one of these becomes true:

- the next deletion would change a public contract without a test proving parity,
- the duplicate modules diverge semantically and need a separate design decision,
- or the code got shorter but harder to understand.

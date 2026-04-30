---
name: diagnose-aria
description: Diagnose ARIA-NBV bugs, regressions, failing metrics, Streamlit issues, docs builds, KG ingestion failures, or performance regressions through a reproducible feedback loop. Use when something is broken, flaky, slow, miscalibrated, or producing suspicious geometry/RRI/VIN outputs.
---

# Diagnose ARIA

## When To Use

Use this skill for hard bugs or regressions in:

- geometry, `PoseTW` / `CameraTW`, rendering, RRI, VIN, or rollout behavior
- immutable VIN offline stores, manifests, split files, or data smoke checks
- Streamlit panels, Quarto / Typst renders, KG ingestion, or CLI failures
- metric drift, calibration collapse, performance regressions, or flaky tests

Do not use it for broad planning without a concrete symptom; use `plan-grill`
for that.

## Feedback Loop First

Build the smallest deterministic loop that reproduces the user-visible symptom:

- package behavior: `cd aria_nbv && uv run pytest <focused-test>`
- CLI/data path: `cd aria_nbv && uv run nbv-summary --config-path <config>`
- immutable store: manifest/sample-index read plus
  `tests/data_handling/test_vin_offline_store.py`
- Streamlit panel: import/dispatcher test before manual UI inspection
- docs: `cd docs && quarto render <page>` or
  `typst compile typst/slides/<file>.typ --root .`
- KG: the narrowest `make kg-*` command that owns the failing artifact
- performance: a timing harness or profiler before changing code

Do not proceed on code guesses until the loop fails in the same way the user
reported, or until you can state exactly why a loop cannot be built.

## Workflow

1. Reproduce the symptom and capture the exact command, traceback, metric, or
   bad output.
2. Minimize the loop until it is fast enough to run repeatedly.
3. Write 3 to 5 ranked falsifiable hypotheses.
4. Probe one variable at a time. Temporary logs must use a unique
   `[DEBUG-...]` prefix.
5. Turn the minimized repro into a regression test when the seam is real.
6. Fix the cause, rerun the minimized loop and original loop, then remove all
   debug instrumentation.

If no good test seam exists, record that as architecture debt in
`.agents/issues.toml` or `.agents/refactors.toml` through `agents-db`.

## ARIA Checks

- Geometry changes should also use `nbv-geometry-contracts`.
- Data/offline-store changes should also use `dataset-cache-ops`.
- Target/entity RRI changes should also use `entity-aware-rri`.
- Rollout/RL failures should also use `counterfactual-rollout-planner`.
- Docs or memory changes should also use `docs-curator`.

## Completion

Report:

- failing loop and passing loop
- confirmed cause, not only the patch
- regression test or reason no correct seam exists
- removed debug probes
- any DB or memory updates needed

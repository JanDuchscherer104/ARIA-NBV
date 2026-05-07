---
name: diagnose-aria
description: Use to diagnose ARIA-NBV bugs, regressions, failing metrics, Streamlit issues, docs builds, KG failures, or suspicious outputs.
metadata:
  mode: diagnostic
  not_when:
    - "broad planning without a concrete symptom"
    - "pure source localization with no failure"
    - "reviewing concrete diffs rather than reproducing a symptom"
  handoff_to:
    - "plan-grill for broad planning without a concrete symptom"
    - "aria-nbv-context for localizing an unknown failure surface"
    - "code-review for diff review"
    - "specialized contract skills after the failing surface is known"
  evidence_required:
    - "smallest reproducible failing loop or explicit reason none exists"
    - "exact command, traceback, metric, or bad output"
    - "passing loop after the fix"
  applies_to:
    - "**"
  triggers:
    - "bug"
    - "regression"
    - "failing test"
    - "suspicious metric"
  must_read:
    - "AGENTS.md"
    - ".agents/memory/state/GOTCHAS.md"
    - ".agents/references/verification_matrix.md"
  verification:
    - "the narrowest reproducer for the failing surface"
    - "focused regression test after fixes"
---

# Diagnose ARIA

## When To Use

Use this skill for hard bugs or regressions in:

- geometry, `PoseTW` / `CameraTW`, rendering, RRI, VIN, or rollout behavior
- immutable VIN offline stores, manifests, split files, or data smoke checks
- Streamlit panels, Quarto / Typst renders, KG ingestion, or CLI failures
- metric drift, calibration collapse, performance regressions, or flaky tests

## Feedback Loop First

Build the smallest deterministic loop that reproduces the user-visible symptom:

- package behavior: `cd aria_nbv && uv run pytest <focused-test>`
- CLI/data path: `cd aria_nbv && uv run nbv-summary --config-path <config>`
- immutable store: manifest/sample-index read plus
  `tests/data_handling/test_vin_offline_store.py`
- Streamlit panel: import/dispatcher test before manual UI inspection
- docs: `cd docs && quarto render <page>` or
  `typst compile typst/seminar_slides/<file>.typ --root .`
- KG: the narrowest `make kg-*` command that owns the failing artifact
- performance: a timing harness or profiler before changing code

Do not proceed on code guesses until the loop fails in the same way the user
reported, or until you can state exactly why a loop cannot be built.

If no reproducible loop can be built, do not patch by guesswork. State the
missing artifact, access, fixture, command, or metric needed next. If the
blocker is durable repo debt, record a blocked issue in `.agents/issues.toml`
through `agents-db`.

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

## Specialized Evidence

After the failing surface is known, include the relevant contract skill named
by root routing or nested package guidance in the repro and verification loop.

## Completion

Report:

- failing loop and passing loop
- confirmed cause, not only the patch
- regression test or reason no correct seam exists
- removed debug probes
- any DB or memory updates needed

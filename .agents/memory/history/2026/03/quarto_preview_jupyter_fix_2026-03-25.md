---
id: 2026-03-25_quarto_preview_jupyter_fix
date: 2026-03-25
title: "Fix Quarto preview Jupyter execution for docs site"
status: done
topics: [docs, quarto, jupyter, environment]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/ext-impl/prj_aria_tools_impl.qmd
artifacts:
  - docs/contents/ext-impl/prj_aria_tools_impl.html
assumptions:
  - The ASE sample dataset under `.data/semidense_samples/ase/ase_examples/0` is optional and should not be required for site preview.
---

## Task

Fix `quarto preview docs/index.qmd --no-browser` after it failed on cached
Jupyter execution and then on a hard assertion inside
`prj_aria_tools_impl.qmd`.

## Method

Repaired the local `python3` kernelspec to point to the NBV venv, installed
`jupyter-cache` into both the NBV venv and Quarto's driver Python
(`/home/jandu/miniforge3/bin/python`), then patched the
`prj_aria_tools_impl.qmd` page so its sample-data examples are skipped cleanly
when the optional ASE example folder is absent.

## Findings

- The `python3` Jupyter kernel was pointing to a stale interpreter in another
  repo (`traenslenzor`), which would have broken notebook execution even after
  the cache issue was resolved.
- Quarto's cached Jupyter execution needs `jupyter-cache` in the Python used by
  Quarto itself, not only in the kernel interpreter.
- `prj_aria_tools_impl.qmd` assumed a local sample-data directory existed and
  aborted the whole site preview with an assertion when it did not.

## Verification

- `quarto render docs/contents/ext-impl/prj_aria_tools_impl.qmd --to html`
- `quarto preview docs/index.qmd --no-browser`
  - verified the preview reached `Watching files for changes` and served on localhost
- `quarto check`

## Canonical State Impact

No canonical state files changed. This was a docs/execution-environment repair.

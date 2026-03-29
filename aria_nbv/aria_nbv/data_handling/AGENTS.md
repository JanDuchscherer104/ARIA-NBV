---
scope: module
applies_to: aria_nbv/aria_nbv/data_handling/**
summary: Raw snippet, oracle cache, and VIN cache contract guidance for work under aria_nbv/aria_nbv/data_handling/.
---

# Data Handling Boundary

Apply this file when working under `aria_nbv/aria_nbv/data_handling/`.

## Public Contracts
- Public package surface: `aria_nbv/aria_nbv/data_handling/__init__.py`
- Raw snippet and typed container surface: `efm_dataset.py`, `efm_views.py`, `efm_snippet_loader.py`
- Cache contracts and index helpers: `cache_contracts.py`, `cache_index.py`, `offline_cache_store.py`, `offline_cache_serialization.py`
- Active cache readers and writers: `oracle_cache.py`, `vin_cache.py`, `vin_adapter.py`, `vin_oracle_datasets.py`, `vin_provider.py`
- Narrative surfaces: `aria_nbv/aria_nbv/data_handling/README.md`, `docs/contents/impl/data_pipeline_overview.qmd`, `docs/typst/paper/sections/12h-appendix-offline-cache.typ`

## Boundary Rules
- `aria_nbv.data_handling` is the active owner of the raw snippet, oracle cache, and VIN cache flow. Treat `aria_nbv.data` as a compatibility surface unless the task explicitly requires backward-compatible changes there.
- Writers own index and split maintenance. Readers should validate or repair via explicit helpers rather than by implicit mutation or manual file edits.
- Do not hand-edit cache metadata, `index.jsonl`, `train_index.jsonl`, `val_index.jsonl`, or sample payloads to silence failing tests; fix the writer, reader, or generator instead.
- Keep one canonical path from `EfmSnippetView` to `VinSnippetView`; do not duplicate VIN-adapter logic in unrelated modules.
- When cache payload, metadata, or split semantics change, update the public surface, docs, and targeted tests together.

## Verification
- Run `ruff format` and `ruff check` on touched data-handling files.
- Run the most direct targeted pytest from `aria_nbv/tests/data_handling/`, plus compatibility or split coverage such as `aria_nbv/tests/data/test_offline_cache_split.py`, `aria_nbv/tests/data/test_vin_snippet_cache.py`, and `aria_nbv/tests/data/test_vin_snippet_cache_datamodule_equivalence.py` when those surfaces are affected.
- Run the relevant Lightning datamodule or integration coverage when the change affects dataset selection, live fallback, or training-facing batch assembly.

## Completion Criteria
- Index, split, and payload semantics are validated by targeted tests.
- No failing check is “fixed” by hand-editing derived cache artifacts.
- Docs reflect any changed cache, snippet, or dataset contract visible outside the module.

---
id: 2026-01-02_vin_v2_frustum_toggle_2026-01-02
date: 2026-01-02
title: "Vin V2 Frustum Toggle 2026 01 02"
status: legacy-imported
topics: [v2, frustum, toggle, 2026, 01]
source_legacy_path: ".codex/vin_v2_frustum_toggle_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

## Summary

- Made the semidense frustum MHCA path optional via `enable_semidense_frustum` (default: `False`).
- Updated VIN v2 documentation to reflect the optional frustum path and the concat list.

## Tests / Validation

- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_utils.py tests/vin/test_vin_model_v2_integration.py`
- `npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg`
- `quarto render docs/contents/impl/aria_nbv_overview.qmd --to html`

## Notes

- When disabled, frustum features are omitted from the head input and the MHCA block is not constructed.

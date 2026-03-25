---
id: 2026-01-07_vin_nbv_flowchart_diagrams_2026-01-07
date: 2026-01-07
title: "Vin Nbv Flowchart Diagrams 2026 01 07"
status: legacy-imported
topics: [flowchart, diagrams, 2026, 01, 07]
source_legacy_path: ".codex/vin_nbv_flowchart_diagrams_2026-01-07.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN-NBV flowchart diagrams (draw.io XML + Mermaid) — 2026-01-07

## Goal

Provide **flowchart-style** architecture diagrams for **VIN-NBV / VIN v2** in:

- **draw.io XML** (`mxfile`) for direct import + manual refinement.
- **Mermaid** for lightweight text-based diagrams + quick PNG renders.

## Outputs

### Sources

- Draw.io XML (multi-page): `docs/diagrams/vin_nbv/drawio/vin_nbv_v2_flowcharts.drawio.xml`
- Mermaid sources (one per diagram): `docs/diagrams/vin_nbv/mermaid/*.mmd`

### Compiled renders (Mermaid → PNG)

- `docs/diagrams/vin_nbv/renders/mermaid/*.png`

Rendered locally via:

```bash
for f in docs/diagrams/vin_nbv/mermaid/*.mmd; do
  npx -y @mermaid-js/mermaid-cli \
    -i "$f" \
    -o "docs/diagrams/vin_nbv/renders/mermaid/$(basename ${f%.mmd}).png"
done
```

## Conventions

- Edge labels carry shapes using LaTeX-style `\\( ... \\)` so they paste cleanly into draw.io with MathJax enabled.
- Mermaid itself can render math via `$$ ... $$` (KaTeX), but this set keeps `\\( ... \\)` as **plain text** for consistency with draw.io exports.

## Mermaid doc notes (Context7)

- Edge labels: `A -->|label| B`
- Node labels with special chars should be quoted.
- Multi-line labels: use `<br/>` inside node labels.

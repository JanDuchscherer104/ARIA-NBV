---
id: 2026-01-06_vin_v2_mermaid_diagrams_2026-01-06
date: 2026-01-06
title: "Vin V2 Mermaid Diagrams 2026 01 06"
status: legacy-imported
topics: [v2, mermaid, diagrams, 2026, 01]
source_legacy_path: ".codex/vin_v2_mermaid_diagrams_2026-01-06.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN v2 Mermaid component diagrams (2026-01-06)

## Goal
Provide Mermaid architecture diagrams for the major components of
`oracle_rri/oracle_rri/vin/model_v2.py` (VIN v2), and validate them locally.

## Deliverables
- `docs/contents/impl/vin_v2_component_diagrams.qmd`
  - 10 Mermaid flowcharts covering:
    1) pose preparation + pose encoder
    2) EVL scene field construction
    3) voxel-valid-fraction computation + gating
    4) pose-conditioned global pooling + MH cross-attention
    5) semidense sampling + projection
    6) semidense projection statistics + FiLM modulation
    7) semidense frustum MHCA
    8) optional PointNeXt semidense encoder + FiLM
    9) optional trajectory encoder + candidate-query attention
    10) feature fusion + CORAL head
- `docs/contents/impl/vin_v2_component_diagrams.html` (rendered via Quarto)

## Validation
- Mermaid blocks rendered individually with mermaid-cli:
  - Extract blocks: `/tmp/vin_v2_mermaid/diagram_*.mmd`
  - Render: `npx -y @mermaid-js/mermaid-cli -i diagram_XX.mmd -o diagram_XX.png`
- Quarto render:
  - `quarto render docs/contents/impl/vin_v2_component_diagrams.qmd --to html`

## Notes / follow-ups
- If we later rename internals in `model_v2.py` (e.g., token feature names or the semidense stats vector),
  update this QMD so diagrams remain a faithful map to the code.

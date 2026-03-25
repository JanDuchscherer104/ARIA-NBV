---
id: 2026-01-07_oracle_rri_flowchart_diagrams_2026-01-07
date: 2026-01-07
title: "Oracle Rri Flowchart Diagrams 2026 01 07"
status: legacy-imported
topics: [flowchart, diagrams, 2026, 01, 07]
source_legacy_path: ".codex/oracle_rri_flowchart_diagrams_2026-01-07.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Oracle RRI pipeline flowcharts (Mermaid) — 2026-01-07

## What changed

Added and revised VIN-style Mermaid flowcharts for the **oracle RRI label generation pipeline**:

`AseEfmDataset -> CandidateViewGenerator -> CandidateDepthRenderer -> build_candidate_pointclouds -> OracleRRI`.

Location:
- `docs/diagrams/oracle_rri/mermaid/` (source `.mmd`)
- `docs/diagrams/oracle_rri/renders/mermaid/` (rendered `.svg` + `.png`)

## Diagrams included

- Pipeline overview: `docs/diagrams/oracle_rri/mermaid/overall.mmd`
- Pipeline overview (compact): `docs/diagrams/oracle_rri/mermaid/overall_compact.mmd`
- Compact per-stage:
  - `docs/diagrams/oracle_rri/mermaid/efm_dataset_compact.mmd`
  - `docs/diagrams/oracle_rri/mermaid/candidate_generation_compact.mmd`
  - `docs/diagrams/oracle_rri/mermaid/candidate_depth_renderer_compact.mmd`
  - `docs/diagrams/oracle_rri/mermaid/candidate_pointclouds_compact.mmd`
  - `docs/diagrams/oracle_rri/mermaid/oracle_rri_compact.mmd`
- Expanded per-stage:
  - `docs/diagrams/oracle_rri/mermaid/efm_dataset.mmd`
  - `docs/diagrams/oracle_rri/mermaid/candidate_generation.mmd` (includes `positional_sampling.py`, `orientations.py`, `candidate_generation_rules.py`)
  - `docs/diagrams/oracle_rri/mermaid/candidate_depth_renderer.mmd`
  - `docs/diagrams/oracle_rri/mermaid/candidate_pointclouds.mmd`
  - `docs/diagrams/oracle_rri/mermaid/oracle_rri.mmd`

## Notation choices (for paper + draw.io)

- SE(3) transforms use `\(\mathbf{T}_{from}^{to}\)` (matches `docs/typst/shared/macros.typ` convention `T(A,B)=T^A_B`, i.e. **A ← B**).
- Paper-aligned symbols:
  - GT mesh: `\(\mathcal{M}_{GT}\)` with faces `\(\mathcal{F}_{GT}\)` (see `docs/typst/shared/macros.typ` and `docs/typst/paper/sections/03-problem-formulation.typ`).
  - Current reconstruction points: `\(\mathcal{P}_t\)`, candidate points: `\(\mathcal{P}_q\)`, fused: `\(\mathcal{P}_{t\cup q}\)`.
  - Depth + intrinsics: `\(D_q\)` and `\(C_q\)` (see `docs/typst/paper/sections/05-oracle-rri.typ`).
  - Distance: `\(\mathrm{CD}(\cdot,\cdot)\)`; oracle label: `\(\mathrm{RRI}(q)\)` with stabilizer `\(+\epsilon\)`.
- Edges carry symbolic “shape-like” annotations in `\(...\)` form (single backslashes), which has been the most robust format for draw.io imports in this repo.

## How to re-render

See `docs/diagrams/oracle_rri/README.md` for the exact `mermaid-cli` commands. The short version:

```bash
for f in docs/diagrams/oracle_rri/mermaid/*.mmd; do
  b=$(basename "$f" .mmd)
  npx -y @mermaid-js/mermaid-cli -i "$f" -o "docs/diagrams/oracle_rri/renders/mermaid/$b.svg"
  npx -y @mermaid-js/mermaid-cli -i "$f" -o "docs/diagrams/oracle_rri/renders/mermaid/$b.png"
done
```

## Open follow-ups / ideas

- If we want **fully editable draw.io pages** (like VIN-NBV), generate a dedicated `mxfile` (`drawio/*.drawio.xml`) with one page per diagram.
- Consider adding a “pipeline orchestration” diagram for `OracleRriLabeler.run(...)` that explicitly shows the `OracleRriLabelBatch` dataclass fields as outputs (currently implicit in `overall.mmd`).

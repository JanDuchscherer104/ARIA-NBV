# VIN v3 mermaid diagram refresh (2026-01-28)

## Scope
- Refreshed all diagrams in `docs/figures/diagrams/vin_nbv/mermaid/` to match the
  i_compact.svg conventions (input green, output red, compute purple; data rectangles).
- Switched all diagrams to Top->Bottom flow and aligned labels to Typst symbol
  conventions (`\mathcal{Q}`, `\mathcal{P}`, `\mathbf{e}`, `\mathbf{g}`, etc.).
- Ensured layer/function names are rendered with `\texttt{...}` and separated
  from symbolic lines with explicit line breaks.

## Files updated
- `docs/figures/diagrams/vin_nbv/mermaid/overall.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/pose_encoder.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/scene_field.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/global_pool.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/semidense_proj.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/semidense_frustum.mmd` (now semidense grid CNN)
- `docs/figures/diagrams/vin_nbv/mermaid/head.mmd`
- `docs/figures/diagrams/vin_nbv/mermaid/trajectory.mmd`

## Notes
- The overall diagram excludes trajectory encoding, per request.
- `semidense_frustum.mmd` now reflects the v3 semidense grid CNN rather than
  frustum attention (kept filename for backward references).
- All diagrams use the same Mermaid config block (ELK layout, htmlLabels, font sizing).

## Follow-ups
- Render via `make mmdc-render MMD_DIR=docs/figures/diagrams/vin_nbv/mermaid`.
- If references expect a renamed semidense grid diagram, consider adding a
  duplicate file with a clearer name while keeping the legacy one.

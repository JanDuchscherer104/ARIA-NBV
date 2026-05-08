# Figures, Tables, And Captions For ARIA-NBV

## Role

Figures and tables are part of the argument. Each one must answer: what should
the reader learn that prose alone would not convey?

## Figure Policy

Use `#figure(...)` with explicit sizing and labels:

```typst
#figure(
  image("figures/vin_offline_store_training.png", width: 92%),
  caption: [Offline training pipeline for the VIN proxy. Logged observations define historical context, counterfactual candidate views are rendered from sampled poses, and oracle RRI labels supervise candidate scoring.],
) <fig:vin-offline-store-training>
```

Captions should contain:

1. object/process being shown;
2. key visual encoding or stages;
3. thesis relevance;
4. no unsupported result claim unless backed by data.

## Table Policy

Use tables for exact values and compact comparisons. Use figures for trends,
architecture, flow, or spatial relationships.

For thesis result tables:

- include metric direction in header or caption;
- report mean and variability when available;
- bold only the best value when it is meaningful;
- do not duplicate all table values in prose;
- reference the conclusion, not the table object.

## Mermaid / Diagram Policy

- Keep `.mmd` as the version-controlled source.
- Render locally with Mermaid CLI or the ARIA-NBV Mermaid workflow.
- Include PNG/SVG/PDF with explicit Typst width.
- Inspect the standalone render and final Typst page.
- Use notation consistent with `docs/typst/shared`.

## Visual QA Checklist

Before merging, check that labels are legible at printed thesis size, nothing
is clipped or blurry, figure width is appropriate, captions wrap cleanly,
cross-references resolve, symbols match shared notation, and page breaks do
not separate figures from necessary explanatory text awkwardly.

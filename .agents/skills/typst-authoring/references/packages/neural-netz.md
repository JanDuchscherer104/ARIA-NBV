# Package: neural-netz

Source: Typst Universe package page.  
Use this package to draw neural network diagrams.

## Quick import
```typst
#import "@preview/neural-netz:0.3.0": draw-network
```

## Core API
`draw-network` is the main entry point:
```typst
#draw-network(
  layers,
  connections: (),
  palette: "warm",
  show-legend: false,
  legend-title: "Layers",
  scale: 100%,
  stroke-thickness: 1,
  depth-multiplier: 0.3,
  show-relu: false,
)
```

## Minimal example
```typst
#draw-network((
  (type: "input", image: "default"),
  (type: "conv", offset: 2),
  (type: "conv", offset: 2),
  (type: "pool"),
  (type: "conv", widths: (1, 1), offset: 3),
))
```
See `references/packages/neural-netz-example.typ` for a minimal standalone example file.

## Connections (skip/aux)
Provide `name` on layers and add `connections`:
```typst
#draw-network((
  (type: "input", label: "A", name: "a", show-connection: true),
  (type: "conv", label: "B", name: "b", offset: 2),
  (type: "conv", label: "C", name: "c", offset: 2),
  (type: "conv", label: "D", name: "d", offset: 2, show-connection: false),
  (type: "conv", label: "E", name: "e", offset: 2),
), connections: (
  (from: "a", to: "c", type: "skip", mode: "depth", label: "depth mode", pos: 6),
  (from: "b", to: "d", type: "skip", mode: "flat", label: "flat mode", pos: 5),
  (from: "c", to: "e", type: "skip", mode: "air", label: "air mode", pos: 5, touch-layer: true),
))
```

## Notes
- Under the hood, `neural-netz` uses the Typst package **CeTZ** for drawing.
- Custom layer type: set `type: "custom"` and override shape/colors.
- Use `show-legend: true` for a smart legend.

## Metadata (from Universe page)
- Current version: **0.3.0**
- Last updated: **2025‑12‑09**
- Minimum Typst version: **0.14.0**
- License: **MIT‑0**
- Repository: https://github.com/edgaremy/neural-netz

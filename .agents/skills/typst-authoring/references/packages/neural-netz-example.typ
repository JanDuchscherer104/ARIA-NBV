#import "@preview/neural-netz:0.3.0": draw-network

= Neural-Netz Examples

== Example (legend + skip connections)

#draw-network(
  (
    (type: "input", label: "Input", name: "in", image: "default", show-connection: true),
    (type: "conv", label: "Conv1", name: "c1", offset: 2),
    (type: "conv", label: "Conv2", name: "c2", offset: 2),
    (type: "pool", label: "Pool", name: "p1"),
    (type: "dense", label: "FC", name: "fc", offset: 2),
    (type: "output", label: "Out", name: "out"),
  ),
  connections: (
    (from: "in", to: "c2", type: "skip", mode: "depth", label: "skip", pos: 5),
    (from: "c1", to: "fc", type: "skip", mode: "air", label: "air", pos: 4, touch-layer: true),
  ),
  show-legend: true,
  legend-title: "Layers",
  palette: "cool",
  scale: 90%,
  stroke-thickness: 1,
  depth-multiplier: 0.3,
  show-relu: true,
)

#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "New Computer Modern")

#let block(pos, label, tint: blue, width: 30mm, ..args) = node(
  pos,
  align(center, label),
  width: width,
  fill: tint.lighten(55%),
  stroke: 1pt + tint.darken(25%),
  corner-radius: 5pt,
  inset: 4pt,
  ..args,
)

#let sum_node(pos, ..args) = node(
  pos,
  text(weight: "bold", 20pt, $+$),
  shape: circle,
  inset: 2pt,
  fill: green.lighten(40%),
  ..args,
)

#let concat_node(pos, ..args) = node(
  pos,
  text(weight: "bold", 18pt, $||$),
  shape: circle,
  inset: 3pt,
  fill: none,
  stroke: 1.6pt,
  ..args,
)

#diagram(
  spacing: (10mm, 8mm),
  cell-size: (10mm, 12mm),
  edge-stroke: 1pt,
  edge-corner-radius: 4pt,
  mark-scale: 70%,

  block((0, 0), [Input], tint: blue, name: <input>),
  block((1, 0), [Conv\ 3x3], tint: green, name: <conv1>),
  block((2, 0), [ReLU], tint: yellow, name: <relu>),
  block((3, 0), [Conv\ 3x3], tint: green, name: <conv2>),
  sum_node((4, 0), name: <sum>),
  block((5, 0), [Output], tint: blue, name: <output>),

  edge(<input>, <conv1>, "-|>"),
  edge(<conv1>, <relu>, "-|>"),
  edge(<relu>, <conv2>, "-|>"),
  edge(<conv2>, <sum>, "-|>"),
  edge(<sum>, <output>, "-|>"),

  edge(<input>, (0, -1), (4, -1), <sum>, "-|>"),
)

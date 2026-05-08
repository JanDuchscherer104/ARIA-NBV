#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
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

#diagram(
  spacing: (12mm, 8mm),
  cell-size: (12mm, 12mm),
  edge-stroke: 1pt,
  edge-corner-radius: 4pt,
  mark-scale: 70%,

  block((0, 0), [Input], tint: blue, name: <input>),
  block((0, 1), [Down\ Block\ 1], tint: green, name: <down1>),
  block((0, 2), [Down\ Block\ 2], tint: green, name: <down2>),

  block((1, 2), [Bottleneck], tint: orange, name: <bottleneck>),

  block((2, 2), [Up\ Block\ 2], tint: purple, name: <up2>),
  block((2, 1), [Up\ Block\ 1], tint: purple, name: <up1>),
  block((2, 0), [Output], tint: blue, name: <output>),

  edge(<input>, <down1>, "-|>"),
  edge(<down1>, <down2>, "-|>"),
  edge(<down2>, <bottleneck>, "-|>"),
  edge(<bottleneck>, <up2>, "-|>"),
  edge(<up2>, <up1>, "-|>"),
  edge(<up1>, <output>, "-|>"),

  edge(<down1>, <up1>, "dashed"),
  edge(<down2>, <up2>, "dashed"),
)

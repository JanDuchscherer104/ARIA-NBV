#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: hexagon, house, parallelogram, diamond, pill

#set page(width: auto, height: auto, margin: 5mm, fill: white)
#set text(font: "New Computer Modern")

#let edge = fletcher.edge

#let blob(
  pos,
  label,
  tint: blue,
  width: 30mm,
  shape: rect,
  ..args,
) = node(
  pos,
  align(center, label),
  width: width,
  fill: tint.lighten(60%),
  stroke: 1pt + tint.darken(20%),
  corner-radius: 5pt,
  inset: 5pt,
  shape: shape,
  ..args,
)

#let input_node(pos, label, tint: purple, width: 32mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: parallelogram.with(angle: 20deg),
  ..args,
)

#let data_node(pos, label, tint: purple, width: 30mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: house.with(angle: 30deg),
  ..args,
)

#let module_node(pos, label, tint: blue, width: 30mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: rect,
  ..args,
)

#let op_node(pos, label, tint: orange, width: 28mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: hexagon,
  ..args,
)

#let act_node(pos, label, tint: yellow, width: 26mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: hexagon,
  ..args,
)

#let decision_node(pos, label, tint: orange, width: 20mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: diamond,
  ..args,
)

#let output_node(pos, label, tint: blue, width: 30mm, ..args) = blob(
  pos,
  label,
  tint: tint,
  width: width,
  shape: pill,
  ..args,
)

#let merge_node(pos, label, tint: green, ..args) = node(
  pos,
  text(weight: "bold", 16pt, label),
  shape: circle,
  inset: 2pt,
  fill: tint.lighten(45%),
  stroke: 1pt + tint.darken(25%),
  ..args,
)

#let diagram_base(..content) = align(center, diagram(
  spacing: 8pt,
  cell-size: (8mm, 10mm),
  edge-stroke: 1pt,
  edge-corner-radius: 5pt,
  mark-scale: 70%,
  ..content,
))

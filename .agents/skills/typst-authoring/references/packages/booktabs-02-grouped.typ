#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

#table(
  columns: 5,
  align: (left, center, center, center, center),
  toprule(),
  table.header(
    [Model],
    table.cell(colspan: 2)[Validation],
    table.cell(colspan: 2)[Test],
  ),
  cmidrule(start: 1, end: 3),
  cmidrule(start: 3, end: 5),
  table.header(
    [],
    [Acc \ %],
    [Loss],
    [Acc \ %],
    [Loss],
  ),
  midrule(),
  [AlexNet],
  [57.1],
  [1.45],
  [57.3],
  [1.44],
  [ResNet-50],
  [76.2],
  [0.82],
  [76.5],
  [0.80],
  [ViT-B/16],
  [77.9],
  [0.76],
  [78.1],
  [0.75],
  bottomrule(),
)

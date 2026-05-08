#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

#table(
  columns: 3,
  align: (left, center, center),
  toprule(),
  table.header(
    [Substance],
    [Subcritical \ #sym.degree C],
    [Supercritical \ #sym.degree C],
  ),
  midrule(),
  [Hydrochloric Acid],
  [12.0],
  [92.1],
  [Potassium Hydroxide],
  table.cell(colspan: 2)[24.7],
  cmidrule(start: 1, end: -1),
  [Sodium Myreth Sulfate],
  [16.6],
  [104],
  bottomrule()
)

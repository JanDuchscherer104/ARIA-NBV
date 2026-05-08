# Common Issues that Codex keeps running into when woring with Typst

- vectors, matrices, and tensors must always be bf.
- prefer bf(cal(...)) for tensors and sets?
- contents after a subscript are often incorrect. typst will render the following op("IoU")_"3D"(hat(bold(B))_(hat(e)), bold(B)_e^"GT") incorrectly - `(hat(bold(B))_(hat(e)), bold(B)_e^"GT")` will render as subscript. to ensure that only "3D" is subscripted, use `op("IoU")_"3D" (hat(bold(B))_(hat(e)), bold(B)_e^"GT")`. So the space between the subscript and following content that is not supposed to be subscripted is important. Please query typst library docs to see how this should optimally be done!
- always ensure usage of symbols from the shared typst symbol and equation library (^1) to ensure consistency across documents. if a symbol or equation is missing, add it to the library and use it from there.
- also consider skills listed in "https://context7.com/skills?q=typst". What helpful guidance from those skills could be added to our skill?

(^1):
```
docs/typst/shared
├── data
│   ├── paper_figures_oracle_labeler.toml
│   ├── vin_offline_store_stats.json
│   ├── vin_v3_01_vs_t41_summary.json
│   ├── wandb_rtjvfyyp_dynamics.json
│   ├── wandb_rtjvfyyp_summary.json
│   └── wandb_top2_improvements.json
├── equations
│   ├── action.typ
│   ├── binning.typ
│   ├── coral.typ
│   ├── coverage.typ
│   ├── entity.typ
│   ├── features.typ
│   ├── metrics.typ
│   ├── rl.typ
│   ├── rri.typ
│   └── vin.typ
├── equations.typ
├── glossary.generated.typ
├── glossary.typ
├── macros.typ
├── math.typ
├── notation.generated.typ
├── slide-template.typ
├── style.typ
├── symbols
│   ├── ase.typ
│   ├── entity.typ
│   ├── frame.typ
│   ├── obs.typ
│   ├── oracle.typ
│   ├── rl.typ
│   ├── shape.typ
│   └── vin.typ
├── symbols.typ
└── terms.typ
```
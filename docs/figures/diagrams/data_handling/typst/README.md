# Typst Data-Layout Trees

This directory contains standalone Typst sources and tracked SVG renders for
hierarchical ARIA-NBV store layouts. They use the shared `tdtr` helpers in
`docs/typst/shared/data-layout-trees.typ`.

`offline_rollout_relation_tree.typ`, `rollout_zarr_tree.typ`, and
`rollout_sample_tree.typ` use connected left-to-right trees so store joins,
table ownership, branch keys, and derived reader views remain visually
connected. The smaller physical-layout trees keep the default top-to-bottom
direction where the hierarchy is shallower.

Render from `docs/`:

```sh
for f in figures/diagrams/data_handling/typst/*_tree.typ; do
  typst compile "$f" "${f%.typ}.svg" --root .
done
```

Use the SVGs from Markdown and import the shared helper directly from Typst
documents that need local styling or sizing.

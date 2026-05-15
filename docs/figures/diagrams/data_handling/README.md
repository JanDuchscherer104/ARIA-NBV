# Data-Handling Diagrams

This package contains the source Mermaid diagrams and rendered SVGs used by
`aria_nbv/aria_nbv/data_handling/README.md`.

- `mermaid/*.mmd` is the editable source.
- `mermaid/*.svg` is the tracked render imported by the package README.
- Keep exact filesystem and table trees in Markdown; use these diagrams for
  data flow, branching, and relational structure.

Render all diagrams from the repository root:

```sh
for f in docs/figures/diagrams/data_handling/mermaid/*.mmd; do
  tools/mermaid/scripts/render_mermaid.sh "$f" "${f%.mmd}.svg"
done
```

Validate source diagrams before committing:

```sh
python tools/mermaid/scripts/aria_mermaid_lint.py docs/figures/diagrams/data_handling/mermaid/*.mmd
```

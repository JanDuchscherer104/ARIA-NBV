# VIN-NBV (VIN v2) flowcharts (draw.io XML + Mermaid)

This folder contains **flowchart-style** architecture diagrams for **VIN-NBV / VIN v2**:

- `drawio/` contains a draw.io-export style `mxfile` (**XML**) with one page per submodule.
- `mermaid/` contains Mermaid equivalents (one `.mmd` file per diagram).

## Render Mermaid diagrams

Use `mermaid-cli` (`mmdc`) to render each file to PNG for quick inspection:

```bash
npx -y @mermaid-js/mermaid-cli \
  -i docs/diagrams/vin_nbv/mermaid/overall.mmd \
  -o docs/diagrams/vin_nbv/renders/mermaid/overall.png
```

Repeat for the other `.mmd` files in `docs/diagrams/vin_nbv/mermaid/`.

## Import the XML into draw.io

Open draw.io (diagrams.net) and import:

- `docs/diagrams/vin_nbv/drawio/vin_nbv_v2_flowcharts.drawio.xml`

The file contains multiple pages (overall + submodules).

## Notes on math in Mermaid vs draw.io

- **draw.io** renders MathJax when enabled and typically uses `\(...\)` / `$$...$$`.
- **Mermaid** has dedicated math support (KaTeX) typically using `$$ ... $$`.

For consistency with the paper + draw.io exports, edge labels in these files use `\(...\)` as “math-like” shape annotations (single backslashes, since that imports more reliably into draw.io).

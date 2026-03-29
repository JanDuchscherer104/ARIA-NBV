# Typst confusion-matrix sequence (2026-01-27)

Summary:
- Replaced hardcoded confusion-matrix frame list with a manifest-driven loader in `docs/typst/slides/template.typ`.
- Frames are parsed from filenames (`confusion_matrix_<step>_<hash>.png`) and ordered by step.
- Integrated the sequence into `docs/typst/slides/slides_4.typ`.
- Set default slide figure caption text size to 14pt via `#show figure.caption`.

Key note:
- Typst cannot enumerate directories at compile time, so the function reads a `frames.json` manifest in the target directory.

Usage:
```
#let cm_dir = "../../figures/vin_v2/val-conf-mats"
#conf-matrix-sequence(
  cm_dir,
  title: [VIN v2 val confusion matrices],
  caption: [Validation confusion matrices over training],
  width: 88%,
)
```

Manifest generation suggestion:
```
python - <<'PY'
from pathlib import Path
import json

base = Path("docs/typst/paper/figures/vin_v2/val-conf-mats")
files = sorted(p.name for p in base.glob("*.png"))
(base / "frames.json").write_text(json.dumps(files, indent=2) + "\n")
PY
```

Compile check:
- `typst compile docs/typst/slides/slides_4.typ` (run after manifest creation).

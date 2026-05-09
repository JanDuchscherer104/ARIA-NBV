#!/usr/bin/env python3
"""Lightweight Typst symbol extractor for docs/typst/shared/symbols/*.typ.

This helper is intentionally simple: it extracts `key: $...$` entries from the
shared Typst symbol modules and emits a YAML-ish report. Human review is still
required to produce Mermaid-compatible KaTeX strings.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ENTRY_RE = re.compile(r"^\s*([a-zA-Z0-9_]+):\s*(\$.*?\$),?\s*$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("typst_shared", type=Path, help="Path to docs/typst/shared")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    symbols_dir = args.typst_shared / "symbols"
    lines: list[str] = ["# Generated draft. Review before use in Mermaid.\n", "symbols:\n"]
    for path in sorted(symbols_dir.glob("*.typ")):
        module = path.stem
        lines.append(f"  {module}:\n")
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = ENTRY_RE.match(raw)
            if m:
                key, typst = m.groups()
                safe_typst = typst.replace("'", "''")
                lines.append(f"    {key}: {{ typst: '{safe_typst}', latex: null, meaning: null }}\n")
    out = "".join(lines)
    if args.output:
        args.output.write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

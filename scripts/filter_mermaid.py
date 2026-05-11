#!/usr/bin/env python3
"""Filter generated Mermaid diagrams by class-name substrings.

`make context-uml` uses this helper to keep syrenka class diagrams small enough
for local context packs. The filter is intentionally syntax-light: it removes
class blocks and relationship lines that mention any excluded substring, while
preserving the surrounding Mermaid header and unrelated nodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_excludes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _line_matches(line: str, excludes: tuple[str, ...]) -> bool:
    return any(token in line for token in excludes)


def filter_mermaid(text: str, excludes: tuple[str, ...]) -> str:
    """Return Mermaid text with excluded class blocks and edges removed."""

    if not excludes:
        return text

    output: list[str] = []
    skipping_class_block = False
    brace_depth = 0

    for line in text.splitlines():
        stripped = line.strip()
        if skipping_class_block:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                skipping_class_block = False
            continue

        if stripped.startswith("class ") and _line_matches(stripped, excludes):
            if "{" in line and "}" not in line:
                skipping_class_block = True
                brace_depth = line.count("{") - line.count("}")
            continue

        if _line_matches(stripped, excludes):
            continue

        output.append(line)

    return "\n".join(output).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input Mermaid file.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered Mermaid output file.")
    parser.add_argument("--exclude", default="", help="Comma-separated substrings to remove.")
    args = parser.parse_args()

    excludes = _parse_excludes(args.exclude)
    text = args.input.read_text(encoding="utf-8")
    args.output.write_text(filter_mermaid(text, excludes), encoding="utf-8")


if __name__ == "__main__":
    main()

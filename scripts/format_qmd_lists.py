#!/usr/bin/env python3
"""Normalize blank lines before list blocks in Quarto sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+\.)\s+")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["docs"], help="Files or directories to rewrite")
    return parser.parse_args()


def iter_qmd_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".qmd":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.qmd")))
    return files


def is_list_item(line: str) -> bool:
    return bool(LIST_ITEM_RE.match(line))


def normalize_qmd(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_marker = ""

    for idx, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        if line.strip() == "":
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            prev_line = out[-1] if out else ""
            if is_list_item(prev_line) and is_list_item(next_line):
                continue
            out.append("")
            continue

        if is_list_item(line):
            prev_line = out[-1] if out else ""
            if prev_line.strip() and not is_list_item(prev_line):
                out.append("")
            out.append(line)
            continue

        out.append(line)

    normalized = "\n".join(out)
    return normalized + ("\n" if text.endswith("\n") or normalized else "")


def main() -> int:
    changed = 0
    for path in iter_qmd_files(parse_args().paths):
        original = path.read_text(encoding="utf-8")
        normalized = normalize_qmd(original)
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")
            changed += 1

    print(f"Normalized {changed} QMD files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

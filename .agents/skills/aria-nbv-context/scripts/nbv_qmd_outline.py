#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _strip_front_matter(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    if lines[0].strip() != "---":
        return lines
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return lines[idx + 1 :]
    return lines


def _iter_headings(lines: list[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        title = re.sub(r"\s+#+\s*$", "", title).strip()
        if title:
            headings.append((level, title))
    return headings


def outline_qmd(path: Path, root: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = _strip_front_matter(text.splitlines())
    headings = _iter_headings(lines)
    if not headings:
        return None
    rel = path.resolve().relative_to(root.resolve())
    out_lines = [f"# {rel.as_posix()}"]
    for level, title in headings:
        indent = "  " * (level - 1)
        out_lines.append(f"{indent}- {title}")
    out_lines.append("")
    return "\n".join(out_lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description="Outline headings in docs/**/*.qmd with nested ordering."
    )
    parser.add_argument("--root", default=str(repo_root / "docs"), help="Docs root dir")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"error: docs root not found: {root}")

    outputs: list[str] = []
    for path in sorted(root.rglob("*.qmd")):
        outline = outline_qmd(path, root)
        if outline:
            outputs.append(outline)

    if outputs:
        print("\n".join(outputs).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

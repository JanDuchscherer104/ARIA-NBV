#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')
HEADING_RE = re.compile(r"^\s*(=+)\s+(.+)$")


def extract_includes(path: Path) -> list[str]:
    includes: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return includes
    for line in text.splitlines():
        match = INCLUDE_RE.match(line)
        if match:
            includes.append(match.group(1))
    return includes


def _iter_headings(lines: list[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for line in lines:
        if line.lstrip().startswith("//"):
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((level, title))
    return headings


def _resolve_include(source: Path, root: Path, inc: str) -> Path | None:
    candidate = (source.parent / inc).resolve()
    if candidate.exists():
        return candidate
    candidate = (root / inc).resolve()
    if candidate.exists():
        return candidate
    return None


def _walk_outline(path: Path, root: Path, stack: list[Path]) -> None:
    if path in stack:
        print(f"warning: include cycle detected: {path}", file=sys.stderr)
        return
    stack.append(path)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        print(f"warning: failed to read: {path}", file=sys.stderr)
        stack.pop()
        return

    rel = path.resolve().relative_to(root.resolve())
    print(f"# {rel.as_posix()}")
    lines = text.splitlines()
    for line in lines:
        match = INCLUDE_RE.match(line)
        if match:
            inc = match.group(1)
            resolved = _resolve_include(path, root, inc)
            if resolved is None:
                print(f"warning: include not found: {inc} (from {path})", file=sys.stderr)
            else:
                _walk_outline(resolved, root, stack)
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            continue
        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        indent = "  " * (level - 1)
        print(f"{indent}- {title}")

    print()
    stack.pop()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / "../../../../").resolve()
    default_root = (repo_root / "docs" / "typst").resolve()

    parser = argparse.ArgumentParser(description="List Typst #include targets for paper/slides")
    parser.add_argument("--root", default=str(default_root), help="Typst root directory")
    parser.add_argument("--paper", default=None, help="Typst paper entry")
    parser.add_argument(
        "--slides", default=None, help="Directory containing slide .typ files"
    )
    parser.add_argument(
        "--mode",
        choices=["graph", "includes"],
        default="graph",
        help="graph: include outline with headings; includes: include list only",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paper = Path(args.paper).resolve() if args.paper else root / "paper" / "main.typ"
    slides_dir = Path(args.slides).resolve() if args.slides else root / "slides"

    targets: list[Path] = []
    if paper.exists():
        targets.append(paper)
    if slides_dir.exists():
        targets.extend(sorted(slides_dir.glob("*.typ")))

    if args.mode == "includes":
        for source in targets:
            rel_source = source.as_posix()
            includes = extract_includes(source)
            if not includes:
                continue
            print(f"# {rel_source}")
            for inc in includes:
                resolved = _resolve_include(source, root, inc)
                if resolved is None:
                    print(f"- {inc} -> (missing)")
                else:
                    print(f"- {inc} -> {resolved.as_posix()}")
            print()
        return 0

    for source in targets:
        _walk_outline(source, root, [])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

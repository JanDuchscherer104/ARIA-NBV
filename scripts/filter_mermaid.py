#!/usr/bin/env python3
"""Filter excluded namespaces or classes from a Mermaid class diagram."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CLASS_RE = re.compile(r"^\s*class\s+(?P<name>\S+)")
NAMESPACE_RE = re.compile(r"^\s*namespace\s+(?P<name>\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Mermaid file")
    parser.add_argument("--output", required=True, help="Output Mermaid file")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated namespace or class prefixes to exclude",
    )
    return parser.parse_args()


def should_exclude(name: str, exclude_tokens: list[str]) -> bool:
    return any(name == token or name.startswith(f"{token}.") for token in exclude_tokens)


def collect_excluded_ids(lines: list[str], exclude_tokens: list[str]) -> set[str]:
    excluded_ids: set[str] = set()
    for line in lines:
        for regex in (CLASS_RE, NAMESPACE_RE):
            match = regex.match(line)
            if not match:
                continue
            name = match.group("name")
            if should_exclude(name, exclude_tokens):
                excluded_ids.add(name)
    return excluded_ids


def filter_lines(lines: list[str], exclude_tokens: list[str], excluded_ids: set[str]) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        if any(token in line for token in exclude_tokens):
            continue
        if any(re.search(rf"\b{re.escape(excluded_id)}\b", line) for excluded_id in excluded_ids):
            continue
        filtered.append(line)
    return filtered


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    exclude_tokens = [token.strip() for token in args.exclude.split(",") if token.strip()]

    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    excluded_ids = collect_excluded_ids(lines, exclude_tokens)
    filtered = filter_lines(lines, exclude_tokens, excluded_ids)

    output_path.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate taxonomy frontmatter for rendered Quarto content pages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = ("title", "phase", "audience", "status", "owner")
VALID_PHASES = {"thesis", "seminar", "archive", "generated"}
VALID_AUDIENCES = {"public", "advisor", "developer", "agent"}
VALID_STATUSES = {"current", "planned", "scratch", "deprecated"}
VALID_OWNERS = {"paper", "docs", "code", "agent", "generated", "jan"}


def _load_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n?", text, flags=re.DOTALL)
    if not match:
        return {}
    data = yaml.safe_load(match.group(1)) or {}
    if not isinstance(data, dict):
        return {}
    return data


def validate(path: Path) -> list[str]:
    metadata = _load_frontmatter(path)
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in metadata:
            errors.append(f"missing `{field}`")

    phase = metadata.get("phase")
    audience = metadata.get("audience")
    status = metadata.get("status")
    owner = metadata.get("owner")

    if phase is not None and phase not in VALID_PHASES:
        errors.append(f"invalid phase `{phase}`")
    if audience is not None and audience not in VALID_AUDIENCES:
        errors.append(f"invalid audience `{audience}`")
    if status is not None and status not in VALID_STATUSES:
        errors.append(f"invalid status `{status}`")
    if owner is not None and owner not in VALID_OWNERS:
        errors.append(f"invalid owner `{owner}`")

    if audience == "agent":
        errors.append("rendered docs/contents pages may not use audience: agent")
    if phase == "generated" and owner != "generated":
        errors.append("phase: generated requires owner: generated")
    if phase == "archive" and status == "scratch":
        errors.append("rendered archive pages must be curated, not scratch")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["docs/contents"],
        help="QMD files or directories to validate.",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.qmd")))
        elif path.suffix == ".qmd":
            files.append(path)

    failures: list[str] = []
    for path in files:
        for error in validate(path):
            failures.append(f"{path}: {error}")

    if failures:
        print("QMD frontmatter validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate agent memory scaffolding and native debrief hygiene.

This checker intentionally stays narrow:

- fail if legacy `.codex/*.md` notes reappear,
- require frontmatter on native debriefs under `.agents/memory/history/`, and
- require the documented native-debrief keys for non-legacy records.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = REPO_ROOT / ".agents" / "memory" / "history"

REQUIRED_NATIVE_KEYS = {
    "id",
    "date",
    "title",
    "status",
    "topics",
    "confidence",
    "canonical_updates_needed",
}


def parse_inline_list(value: str) -> list[str]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]


def parse_frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")

    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("unterminated YAML frontmatter")

    payload = parts[0].removeprefix("---\n")
    data: dict[str, object] = {}
    current_key: str | None = None

    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue

        key_match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", raw_line)
        if key_match:
            key = key_match.group(1)
            value = key_match.group(2).strip()

            if value == "":
                data[key] = []
                current_key = key
                continue

            if value.startswith("[") and value.endswith("]"):
                data[key] = parse_inline_list(value)
            else:
                data[key] = value.strip("\"'")
            current_key = None
            continue

        list_match = re.match(r"^\s*-\s+(.*)$", raw_line)
        if list_match and current_key is not None:
            current_value = data.get(current_key)
            if not isinstance(current_value, list):
                raise ValueError(f"`{current_key}` must be a list when using list items")
            current_value.append(list_match.group(1).strip().strip("\"'"))
            continue

        if current_key is not None and raw_line.startswith(" "):
            # Allow nested metadata under list items (for example `files_touched`
            # entries with `path` / `kind`). The validator does not need to
            # interpret that structure.
            continue

        raise ValueError(f"unsupported frontmatter line: {raw_line!r}")

    return data


def check_codex_notes() -> list[str]:
    codex_dir = REPO_ROOT / ".codex"
    if not codex_dir.exists():
        return []

    notes = sorted(path.relative_to(REPO_ROOT).as_posix() for path in codex_dir.rglob("*.md"))
    if not notes:
        return []

    errors = ["legacy `.codex/*.md` notes are not allowed:"] + [f"  - {note}" for note in notes]
    return errors


def check_history_records() -> list[str]:
    errors: list[str] = []
    if not HISTORY_ROOT.exists():
        return [f"missing history root: {HISTORY_ROOT.relative_to(REPO_ROOT).as_posix()}"]

    for path in sorted(HISTORY_ROOT.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            frontmatter = parse_frontmatter(path)
        except ValueError as exc:
            errors.append(f"{rel}: {exc}")
            continue

        status = str(frontmatter.get("status", "")).strip()
        if status == "legacy-imported":
            continue

        missing_keys = sorted(REQUIRED_NATIVE_KEYS - frontmatter.keys())
        if missing_keys:
            errors.append(f"{rel}: missing required frontmatter keys: {', '.join(missing_keys)}")
            continue

        canonical_updates = frontmatter.get("canonical_updates_needed")
        if not isinstance(canonical_updates, list):
            errors.append(f"{rel}: `canonical_updates_needed` must be a list or []")
            continue

        for update_path in canonical_updates:
            update_text = str(update_path).strip()
            if not update_text:
                errors.append(f"{rel}: empty path in `canonical_updates_needed`")
                continue
            resolved = REPO_ROOT / update_text
            if not resolved.exists():
                errors.append(f"{rel}: canonical update path does not exist: {update_text}")

    return errors


def main() -> int:
    errors = [*check_codex_notes(), *check_history_records()]
    if not errors:
        print("agent memory validation passed")
        return 0

    print("agent memory validation failed", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

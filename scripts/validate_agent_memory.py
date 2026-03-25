#!/usr/bin/env python3
"""Validate agent memory notes and block ad hoc `.codex` markdown files."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = REPO_ROOT / ".agents" / "memory" / "history"
CODEX_ROOT = REPO_ROOT / ".codex"
FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z0-9_]+):(?:\s*(.*))?$")
LIST_ITEM_RE = re.compile(r"^\s*-\s+(.*)$")
LEGACY_STATUS = "legacy-imported"
BASE_REQUIRED_FIELDS = {
    "id",
    "date",
    "title",
    "status",
    "topics",
    "confidence",
}
NATIVE_REQUIRED_FIELDS = BASE_REQUIRED_FIELDS | {"canonical_updates_needed"}


def read_text(path: Path) -> str:
    """Read a text file using tolerant UTF-8 decoding."""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_frontmatter(text: str) -> str | None:
    """Return the leading YAML frontmatter block, if present."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    return text[4:end]


def parse_frontmatter(frontmatter: str) -> dict[str, str | list[str]]:
    """Parse a small frontmatter subset needed for scaffold validation."""
    data: dict[str, str | list[str]] = {}
    current_key: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            match = FRONTMATTER_KEY_RE.match(line)
            if not match:
                current_key = None
                continue
            current_key = match.group(1)
            data[current_key] = (match.group(2) or "").strip()
            continue
        if current_key is None:
            continue
        list_match = LIST_ITEM_RE.match(line)
        if not list_match:
            continue
        items_key = f"{current_key}.__items__"
        items = data.setdefault(items_key, [])
        if not isinstance(items, list):
            msg = f"Expected list storage for frontmatter field {current_key!r}"
            raise TypeError(msg)
        items.append(list_match.group(1).strip())

    return data


def field_present(data: dict[str, str | list[str]], key: str) -> bool:
    """Return whether a required frontmatter field is meaningfully present."""
    if key not in data:
        return False
    value = data[key]
    if isinstance(value, list):
        return bool(value)
    if value.strip() in {"", "null"}:
        return bool(data.get(f"{key}.__items__"))
    return True


def validate_codex_markdown() -> list[str]:
    """Report markdown notes that still live under `.codex/`."""
    if not CODEX_ROOT.exists():
        return []
    return [
        f"Ad hoc Codex note must be migrated or removed: {path.relative_to(REPO_ROOT)}"
        for path in sorted(CODEX_ROOT.rglob("*.md"))
    ]


def validate_history_file(path: Path) -> list[str]:
    """Validate one history note against the native or legacy schema."""
    errors: list[str] = []
    text = read_text(path)
    frontmatter = extract_frontmatter(text)
    rel = path.relative_to(REPO_ROOT)

    if frontmatter is None:
        errors.append(f"Missing YAML frontmatter: {rel}")
        return errors

    data = parse_frontmatter(frontmatter)
    status = str(data.get("status", "")).strip()
    required_fields = (
        BASE_REQUIRED_FIELDS if status == LEGACY_STATUS else NATIVE_REQUIRED_FIELDS
    )

    errors.extend(
        f"Missing required field `{field}` in {rel}"
        for field in sorted(required_fields)
        if not field_present(data, field)
    )

    return errors


def main() -> int:
    """Validate the repo's agent-memory scaffold and return a shell exit code."""
    errors = validate_codex_markdown()

    if HISTORY_ROOT.exists():
        for path in sorted(HISTORY_ROOT.rglob("*.md")):
            errors.extend(validate_history_file(path))

    if errors:
        print("Agent memory validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Agent memory validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

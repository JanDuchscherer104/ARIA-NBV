#!/usr/bin/env python3
"""Validate NBV agent scaffold health beyond debrief hygiene."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import validate_agent_memory

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
AGENTS_DIR = REPO_ROOT / ".agents"

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

SCAN_FILES = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "aria_nbv" / "AGENTS.md",
    REPO_ROOT / "docs" / "AGENTS.md",
    REPO_ROOT / ".agents" / "AGENTS_INTERNAL_DB.md",
    REPO_ROOT / ".agents" / "references" / "codex_hooks.md",
    REPO_ROOT / ".agents" / "references" / "gitnexus_optional.md",
]

PATH_PREFIXES = (
    ".agents/",
    ".codex/",
    ".github/",
    "AGENTS.md",
    "Makefile",
    "README.md",
    "aria_nbv/",
    "docs/",
    "notebooks/",
    "scripts/",
)

REQUIRED_DB_FIELDS = {
    "issues": {
        "id",
        "title",
        "status",
        "priority",
        "category",
        "summary",
        "files",
        "notes",
        "sources",
    },
    "todos": {
        "id",
        "title",
        "status",
        "priority",
        "issue_ids",
        "loc_min",
        "loc_expected",
        "loc_max",
        "files",
        "acceptance",
        "sources",
    },
}


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def _strip_fragment(target: str) -> str:
    return target.split("#", 1)[0]


def _candidate_from_backtick(text: str) -> str | None:
    value = text.strip().strip(".,;:")
    if not value.startswith(PATH_PREFIXES):
        return None
    if any(char in value for char in (" ", "\t", "\n", "<", ">", "{", "}", "|", "$")):
        return None
    if value.endswith((".py:", ".md:")):
        return None
    return _strip_fragment(value)


def _resolve_target(source_file: Path, target: str) -> Path | None:
    clean = _strip_fragment(target.strip())
    if not clean or _is_external(clean):
        return None
    if clean.startswith("/"):
        return None
    return (source_file.parent / clean).resolve()


def _scaffold_markdown_files() -> list[Path]:
    tracked = {
        REPO_ROOT / line
        for line in subprocess.check_output(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    }
    files = [path for path in SCAN_FILES if path in tracked]
    files.extend(path for path in tracked if path.match("aria_nbv/**/AGENTS.md"))
    files.extend(path for path in tracked if path.match("docs/**/AGENTS.md"))
    files.extend(path for path in tracked if path.match(".agents/skills/*/SKILL.md"))
    files.extend(path for path in tracked if path.match(".agents/references/*.md"))
    files.extend(path for path in tracked if path.match(".agents/skills/*/references/*.md"))
    return sorted({path for path in files if path.exists()})


def check_markdown_paths() -> list[str]:
    """Validate local markdown links and simple backticked scaffold paths."""
    errors: list[str] = []
    for source_file in _scaffold_markdown_files():
        text = source_file.read_text(encoding="utf-8")
        rel_source = source_file.relative_to(REPO_ROOT).as_posix()

        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group(1)
            resolved = _resolve_target(source_file, target)
            if resolved is not None and not resolved.exists():
                errors.append(f"{rel_source}: missing markdown link target: {target}")

        for match in BACKTICK_RE.finditer(text):
            candidate = _candidate_from_backtick(match.group(1))
            if candidate is None:
                continue
            resolved = (REPO_ROOT / candidate).resolve()
            if not resolved.exists():
                errors.append(f"{rel_source}: missing referenced path: {candidate}")
    return errors


def check_skill_frontmatter() -> list[str]:
    """Validate required skill metadata."""
    errors: list[str] = []
    skill_paths = [
        REPO_ROOT / line
        for line in subprocess.check_output(
            ["git", "ls-files", ".agents/skills/*/SKILL.md"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    ]
    for skill_path in sorted(skill_paths):
        rel = skill_path.relative_to(REPO_ROOT).as_posix()
        text = skill_path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        if not match:
            errors.append(f"{rel}: missing YAML frontmatter")
            continue
        fields: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
        for required in ("name", "description"):
            if not fields.get(required):
                errors.append(f"{rel}: missing `{required}` in frontmatter")
    return errors


def _load_db(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _check_record_fields(path: Path, collection: str, records: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list):
        errors.append(
            f"{path.relative_to(REPO_ROOT).as_posix()}: `{collection}` must be a list"
        )
        return errors
    required = REQUIRED_DB_FIELDS.get(collection, set())
    for record in records:
        if not isinstance(record, dict):
            errors.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: `{collection}` entry must be a table"
            )
            continue
        record_id = str(record.get("id", "<unknown>"))
        missing = sorted(required - record.keys())
        if missing:
            errors.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: {record_id} missing fields: {', '.join(missing)}"
            )
        if collection == "todos":
            loc_values = [
                record.get("loc_min"),
                record.get("loc_expected"),
                record.get("loc_max"),
            ]
            if not all(isinstance(value, int) for value in loc_values):
                errors.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: {record_id} LOC fields must be integers"
                )
            elif not (0 <= loc_values[0] <= loc_values[1] <= loc_values[2]):
                errors.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: {record_id} LOC fields must be ordered"
                )
    return errors


def check_agent_db() -> list[str]:
    """Validate the active/resolved agent DB TOML files."""
    errors: list[str] = []
    for path in (
        AGENTS_DIR / "issues.toml",
        AGENTS_DIR / "todos.toml",
        AGENTS_DIR / "resolved.toml",
    ):
        if not path.exists():
            errors.append(
                f"missing agent DB file: {path.relative_to(REPO_ROOT).as_posix()}"
            )
            continue
        try:
            data = _load_db(path)
        except tomllib.TOMLDecodeError as exc:
            errors.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: invalid TOML: {exc}"
            )
            continue
        if not isinstance(data.get("meta"), dict):
            errors.append(f"{path.relative_to(REPO_ROOT).as_posix()}: missing [meta]")
        if path.name == "issues.toml":
            errors.extend(_check_record_fields(path, "issues", data.get("issues", [])))
        if path.name == "todos.toml":
            errors.extend(_check_record_fields(path, "todos", data.get("todos", [])))
        if path.name == "resolved.toml":
            for collection in ("resolved_issues", "resolved_todos"):
                records = data.get(collection, [])
                if not isinstance(records, list):
                    errors.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}: `{collection}` must be a list"
                    )
    return errors


def check_expected_scripts() -> list[str]:
    """Validate helper scripts advertised by Makefile and scaffold docs."""
    required = [
        "scripts/nbv_context_index.sh",
        "scripts/nbv_get_context.sh",
        "scripts/nbv_literature_index.sh",
        "scripts/nbv_literature_search.sh",
        "scripts/nbv_qmd_outline.sh",
        "scripts/nbv_typst_includes.py",
        "scripts/quarto_generate_agent_docs.py",
        "scripts/validate_agent_memory.py",
        "scripts/validate_agent_scaffold.py",
        ".agents/scripts/agents_db.py",
        ".codex/config.toml",
        ".codex/hooks.json",
        ".codex/hooks/run_make_context_clean.sh",
    ]
    return [
        f"missing expected helper: {path}"
        for path in required
        if not (REPO_ROOT / path).exists()
    ]


def check_codex_hooks() -> list[str]:
    """Validate active Codex hook config files."""
    errors: list[str] = []
    config_path = REPO_ROOT / ".codex" / "config.toml"
    hooks_path = REPO_ROOT / ".codex" / "hooks.json"
    try:
        config = _load_db(config_path)
    except tomllib.TOMLDecodeError as exc:
        errors.append(f".codex/config.toml: invalid TOML: {exc}")
    else:
        features = config.get("features", {})
        if not isinstance(features, dict) or features.get("codex_hooks") is not True:
            errors.append(".codex/config.toml: expected [features].codex_hooks = true")

    try:
        data = json.loads(hooks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f".codex/hooks.json: invalid JSON: {exc}")
    else:
        hooks = data.get("hooks")
        if not isinstance(hooks, dict) or "SessionStart" not in hooks:
            errors.append(".codex/hooks.json: expected a SessionStart hook")
    return errors


def main() -> int:
    errors = [
        *validate_agent_memory.check_codex_notes(),
        *validate_agent_memory.check_history_records(),
        *check_expected_scripts(),
        *check_markdown_paths(),
        *check_skill_frontmatter(),
        *check_agent_db(),
        *check_codex_hooks(),
    ]
    if not errors:
        print("agent scaffold validation passed")
        return 0

    print("agent scaffold validation failed", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

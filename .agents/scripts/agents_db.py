#!/usr/bin/env python3
"""Manage the local NBV .agents issue and todo databases."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

Kind = Literal["issue", "todo"]
Format = Literal["text", "json"]

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / ".agents"
ISSUES_PATH = AGENTS_DIR / "issues.toml"
TODOS_PATH = AGENTS_DIR / "todos.toml"
RESOLVED_PATH = AGENTS_DIR / "resolved.toml"

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
ISSUE_STATUS_ORDER = {"open": 3, "in_progress": 2, "blocked": 1, "closed": 0}
TODO_STATUS_ORDER = {"pending": 3, "in_progress": 2, "blocked": 1, "done": 0}


def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML document from disk."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _format_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_toml_string(value)]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, list):
        if not value:
            return ["[]"]
        lines = ["["]
        for item in value:
            lines.append(f"    {_toml_string(str(item))},")
        lines.append("]")
        return lines
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _append_assignment(lines: list[str], key: str, value: Any) -> None:
    rendered = _format_value(value)
    if len(rendered) == 1:
        lines.append(f"{key} = {rendered[0]}")
        return
    lines.append(f"{key} = {rendered[0]}")
    lines.extend(rendered[1:])


def dump_toml(path: Path, data: dict[str, Any]) -> None:
    """Write the limited TOML shape used by the local agent DB."""
    lines: list[str] = []
    meta = data.get("meta")
    if isinstance(meta, dict):
        lines.append("[meta]")
        for key, value in meta.items():
            _append_assignment(lines, key, value)

    for table in ("issues", "todos", "resolved_issues", "resolved_todos"):
        records = data.get(table, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if lines:
                lines.append("")
            lines.append(f"[[{table}]]")
            for key, value in record.items():
                _append_assignment(lines, key, value)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _priority_value(priority: str) -> int:
    return PRIORITY_ORDER.get(priority, 0)


def _issue_status_value(status: str) -> int:
    return ISSUE_STATUS_ORDER.get(status, 0)


def _todo_status_value(status: str) -> int:
    return TODO_STATUS_ORDER.get(status, 0)


def validate_todo_record(todo: dict[str, Any]) -> None:
    """Validate required LOC estimates on a todo record."""
    required_fields = ("loc_min", "loc_expected", "loc_max")
    missing_fields = [field for field in required_fields if field not in todo]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Todo {todo.get('id', '<unknown>')} is missing required LOC fields: {missing}"
        )

    loc_min = todo["loc_min"]
    loc_expected = todo["loc_expected"]
    loc_max = todo["loc_max"]
    if not all(isinstance(value, int) for value in (loc_min, loc_expected, loc_max)):
        raise ValueError(
            f"Todo {todo.get('id', '<unknown>')} must use integer LOC estimates."
        )
    if not (0 <= loc_min <= loc_expected <= loc_max):
        raise ValueError(
            f"Todo {todo.get('id', '<unknown>')} must satisfy 0 <= loc_min <= loc_expected <= loc_max."
        )


def rank_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active issues ordered by priority and status."""
    return sorted(
        issues,
        key=lambda issue: (
            -_priority_value(str(issue.get("priority", ""))),
            -_issue_status_value(str(issue.get("status", ""))),
            str(issue.get("id", "")),
        ),
    )


def rank_todos(todos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active todos ordered by priority, status, and expected LOC."""
    for todo in todos:
        validate_todo_record(todo)
    return sorted(
        todos,
        key=lambda todo: (
            -_priority_value(str(todo.get("priority", ""))),
            -_todo_status_value(str(todo.get("status", ""))),
            int(todo["loc_expected"]),
            int(todo["loc_min"]),
            str(todo.get("id", "")),
        ),
    )


def build_ranked_view() -> dict[str, list[dict[str, Any]]]:
    """Build the ranked active backlog view."""
    issues_data = load_toml(ISSUES_PATH)
    todos_data = load_toml(TODOS_PATH)
    return {
        "issues": rank_issues(list(issues_data.get("issues", []))),
        "todos": rank_todos(list(todos_data.get("todos", []))),
    }


def render_ranked_text(
    kind: str, ranked: dict[str, list[dict[str, Any]]], limit: int | None
) -> str:
    """Render ranked backlog data in a compact text format."""
    lines: list[str] = []
    if kind in {"issues", "all"}:
        lines.append("Issues")
        issues = ranked["issues"][:limit] if limit is not None else ranked["issues"]
        for index, issue in enumerate(issues, start=1):
            lines.append(
                f"{index}. {issue['id']} [{issue['priority']}/{issue['status']}] {issue['title']}"
            )
            lines.append(f"   {issue['summary']}")

    if kind in {"todos", "all"}:
        if lines:
            lines.append("")
        lines.append("Todos")
        todos = ranked["todos"][:limit] if limit is not None else ranked["todos"]
        for index, todo in enumerate(todos, start=1):
            loc_triplet = f"{todo['loc_min']}/{todo['loc_expected']}/{todo['loc_max']}"
            linked_issues = ", ".join(todo.get("issue_ids", []))
            lines.append(
                f"{index}. {todo['id']} [{todo['priority']}/{todo['status']}] loc={loc_triplet} {todo['title']}"
            )
            lines.append(f"   issues={linked_issues}")
    return "\n".join(lines)


def _active_collection_key(kind: Kind) -> str:
    return "issues" if kind == "issue" else "todos"


def _resolved_collection_key(kind: Kind) -> str:
    return "resolved_issues" if kind == "issue" else "resolved_todos"


def _active_path(kind: Kind) -> Path:
    return ISSUES_PATH if kind == "issue" else TODOS_PATH


def resolve_record(
    kind: Kind, record_id: str, note: str, resolved_on: str | None = None
) -> dict[str, Any]:
    """Move an active issue or todo into the resolved collection."""
    resolved_stamp = resolved_on or date.today().isoformat()
    active_path = _active_path(kind)
    active_data = load_toml(active_path)
    resolved_data = load_toml(RESOLVED_PATH)

    active_key = _active_collection_key(kind)
    resolved_key = _resolved_collection_key(kind)
    active_records = list(active_data.get(active_key, []))
    resolved_records = list(resolved_data.get(resolved_key, []))

    record_index = next(
        (
            index
            for index, record in enumerate(active_records)
            if record.get("id") == record_id
        ),
        None,
    )
    if record_index is None:
        raise ValueError(f"Could not find {kind} {record_id} in {active_path}.")

    record = dict(active_records.pop(record_index))
    record["previous_status"] = str(record.get("status", ""))
    record["status"] = "resolved"
    record["resolved_on"] = resolved_stamp
    record["resolution_note"] = note
    record["source_collection"] = active_key
    resolved_records.append(record)

    active_data[active_key] = active_records
    active_data.setdefault("meta", {})["updated_on"] = resolved_stamp
    resolved_data[resolved_key] = resolved_records
    resolved_data.setdefault("meta", {})["updated_on"] = resolved_stamp

    dump_toml(active_path, active_data)
    dump_toml(RESOLVED_PATH, resolved_data)
    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the local NBV .agents issue and todo DB."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser(
        "rank", help="Show ranked active issues and todos."
    )
    rank_parser.add_argument(
        "--kind", choices=("issues", "todos", "all"), default="all"
    )
    rank_parser.add_argument("--format", choices=("text", "json"), default="text")
    rank_parser.add_argument("--limit", type=int, default=None)

    resolve_parser = subparsers.add_parser(
        "resolve", help="Move an issue or todo into the resolved collection."
    )
    resolve_parser.add_argument("kind", choices=("issue", "todo"))
    resolve_parser.add_argument("record_id")
    resolve_parser.add_argument("--note", required=True, help="Short resolution note.")
    resolve_parser.add_argument(
        "--resolved-on", default=None, help="Resolution date in YYYY-MM-DD format."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the agent DB CLI."""
    args = _parse_args(argv)

    if args.command == "rank":
        ranked = build_ranked_view()
        output_format: Format = args.format
        if output_format == "json":
            payload = ranked if args.kind == "all" else {args.kind: ranked[args.kind]}
            print(json.dumps(payload, indent=2))
        else:
            print(render_ranked_text(args.kind, ranked, args.limit))
        return 0

    if args.command == "resolve":
        kind: Kind = args.kind
        record = resolve_record(kind, args.record_id, args.note, args.resolved_on)
        print(f"Moved {kind} {record['id']} to {RESOLVED_PATH}.")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

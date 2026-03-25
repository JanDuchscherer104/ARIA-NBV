#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / ".codex"
STATE_DIR = REPO_ROOT / ".agents" / "memory" / "state"
HISTORY_DIR = REPO_ROOT / ".agents" / "memory" / "history"
INDEX_DIR = REPO_ROOT / ".agents" / "memory" / "index"
ARCHIVE_DIR = REPO_ROOT / "archive" / "codex-legacy"
MANIFEST_PATH = INDEX_DIR / "codex_migration_manifest.md"
MIGRATION_DATE = "2026-03-24"

CANONICAL_INPUTS = {
    "AGENTS.md",
    "AGENTS_INTERNAL_DB.md",
    "AGENTS-paper-slides.md",
    "academic-writing-guidelines.md",
    "agents-paper-slides-notes.md",
    "agents_md_improvements.md",
    "all_memories_merged.md",
    "notes.md",
    "past-conversation.md",
}
GENERATED_NOTES = {"codex_make_context.md", "context_sources_index.md"}
STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "review",
    "report",
    "update",
    "fix",
    "notes",
    "paper",
    "slides",
    "codex",
    "nbv",
    "vin",
    "oracle",
    "rri",
}
DATE_PATTERNS = [
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})"),
    re.compile(r"(?P<date>\d{4}\+\d{2}\+\d{2})"),
]
PREFIX_PATTERNS = [
    re.compile(r"^\d{4}[+_-]\d{2}[+_-]\d{2}[+_-]*"),
    re.compile(r"^\d{4}-\d{2}-\d{2}[+_-]*"),
]


@dataclass(slots=True)
class MigrationRecord:
    source: Path
    category: str
    target: Path | None
    date: str | None


def git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def parse_date(name: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            return match.group("date").replace("+", "-")
    return None


def strip_date_prefix(stem: str) -> str:
    value = stem
    for pattern in PREFIX_PATTERNS:
        value = pattern.sub("", value)
    value = value.strip("_+-")
    return value or stem


def slug_tokens(stem: str) -> list[str]:
    base = strip_date_prefix(stem)
    tokens = [token.lower() for token in re.split(r"[^a-zA-Z0-9]+", base) if token]
    return [token for token in tokens if token not in STOPWORDS][:5]


def title_from_stem(stem: str) -> str:
    base = strip_date_prefix(stem)
    parts = [part for part in re.split(r"[_+-]+", base) if part]
    return " ".join(part.capitalize() for part in parts) or stem


def classify(path: Path) -> MigrationRecord:
    name = path.name
    if name in GENERATED_NOTES:
        return MigrationRecord(path, "generated", None, None)
    if name in CANONICAL_INPUTS:
        return MigrationRecord(
            path, "canonical_input", ARCHIVE_DIR / "canonical-inputs" / name, None
        )

    date = parse_date(name)
    if date is not None:
        stem = strip_date_prefix(path.stem)
        target = HISTORY_DIR / date[:4] / date[5:7] / f"{stem or path.stem}.md"
        return MigrationRecord(path, "history", target, date)

    return MigrationRecord(path, "archive", ARCHIVE_DIR / "flat" / name, None)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if git_tracked(src):
        run_git("mv", str(src.relative_to(REPO_ROOT)), str(dst.relative_to(REPO_ROOT)))
    else:
        shutil.move(src, dst)


def render_history_note(record: MigrationRecord) -> str:
    source = record.source.read_text(encoding="utf-8", errors="ignore").strip()
    topics = slug_tokens(record.source.stem)
    topics_yaml = ", ".join(topics)
    title = title_from_stem(record.source.stem)
    body = source or "No note body was preserved in the legacy file."
    return (
        "---\n"
        f"id: {record.date}_{record.source.stem}\n"
        f"date: {record.date}\n"
        f'title: "{title}"\n'
        "status: legacy-imported\n"
        f"topics: [{topics_yaml}]\n"
        f'source_legacy_path: ".codex/{record.source.name}"\n'
        "confidence: low\n"
        "---\n\n"
        f"> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.\n\n"
        f"{body}\n"
    )


def write_history_record(record: MigrationRecord) -> None:
    assert record.target is not None
    ensure_parent(record.target)
    content = render_history_note(record)
    if git_tracked(record.source):
        run_git("rm", "-f", str(record.source.relative_to(REPO_ROOT)))
    else:
        record.source.unlink()
    record.target.write_text(content, encoding="utf-8")
    run_git("add", str(record.target.relative_to(REPO_ROOT)))


def write_manifest(records: list[MigrationRecord]) -> None:
    ensure_parent(MANIFEST_PATH)
    canonical = [record for record in records if record.category == "canonical_input"]
    history = [record for record in records if record.category == "history"]
    archive = [record for record in records if record.category == "archive"]
    generated = [record for record in records if record.category == "generated"]

    lines = [
        "# Codex Migration Manifest",
        "",
        f"- Generated: {MIGRATION_DATE}",
        "- Source: `.codex/*.md` flat note inventory",
        "",
        "## Summary",
        f"- Canonical inputs archived: {len(canonical)}",
        f"- History records imported: {len(history)}",
        f"- Legacy archive records: {len(archive)}",
        f"- Generated notes removed from repo guidance paths: {len(generated)}",
        "",
        "## Canonical Inputs",
    ]
    lines.extend(
        f"- `{record.source.name}` -> `{record.target.relative_to(REPO_ROOT)}`"
        for record in canonical
    )
    lines.append("")
    lines.append("## History Imports")
    lines.extend(
        f"- `{record.source.name}` -> `{record.target.relative_to(REPO_ROOT)}`"
        for record in history
    )
    lines.append("")
    lines.append("## Legacy Archive")
    lines.extend(
        f"- `{record.source.name}` -> `{record.target.relative_to(REPO_ROOT)}`"
        for record in archive
    )
    lines.append("")
    lines.append("## Removed Generated Notes")
    lines.extend(f"- `{record.source.name}`" for record in generated)
    lines.append("")
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def migrate(apply: bool) -> None:
    legacy_notes = sorted(path for path in LEGACY_DIR.glob("*.md"))
    records = [classify(path) for path in legacy_notes]
    write_manifest(records)
    if not apply:
        return

    run_git("add", str(MANIFEST_PATH.relative_to(REPO_ROOT)))
    for record in records:
        if record.category == "generated":
            if git_tracked(record.source):
                run_git("rm", "-f", str(record.source.relative_to(REPO_ROOT)))
            elif record.source.exists():
                record.source.unlink()
            continue
        if record.category == "history":
            write_history_record(record)
            continue
        if record.target is None:
            continue
        move_path(record.source, record.target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate flat .codex notes into agent-memory and archive locations."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration instead of only writing the manifest.",
    )
    args = parser.parse_args()
    migrate(apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

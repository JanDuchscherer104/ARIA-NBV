#!/usr/bin/env python3
"""Validate agent memory records and scaffold routing surfaces."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT_DIR / ".agents" / "memory" / "history"
SOURCE_INDEX = ROOT_DIR / "docs" / "_generated" / "context" / "source_index.md"
LITERATURE_TEX_SRC = ROOT_DIR / "docs" / "literature" / "tex-src"
AGENTS_SEARCH_ROOTS = (
    ROOT_DIR / "aria_nbv",
    ROOT_DIR / "docs",
)
ROOT_AGENTS_FILE = ROOT_DIR / "AGENTS.md"
BOUNDARY_GUIDE_ROOTS = (
    ROOT_DIR / "aria_nbv" / "aria_nbv",
    ROOT_DIR / "docs" / "typst" / "paper",
)
SOURCE_INDEX_PATH_LOCAL_GUIDES = (
    ROOT_DIR / "aria_nbv" / "AGENTS.md",
    ROOT_DIR / "aria_nbv" / "aria_nbv" / "vin" / "AGENTS.md",
    ROOT_DIR / "aria_nbv" / "aria_nbv" / "data_handling" / "AGENTS.md",
    ROOT_DIR / "aria_nbv" / "aria_nbv" / "rri_metrics" / "AGENTS.md",
    ROOT_DIR / "docs" / "AGENTS.md",
    ROOT_DIR / "docs" / "typst" / "paper" / "AGENTS.md",
)

NON_AGENT_SCAFFOLD_FILES = [
    ROOT_DIR / "Makefile",
    ROOT_DIR / ".agents" / "memory" / "README.md",
    ROOT_DIR / ".agents" / "references" / "tooling_skill_governance.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "PROJECT_STATE.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "OWNER_DIRECTIVES.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "DECISIONS.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "OPEN_QUESTIONS.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "GOTCHAS.md",
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "SKILL.md",
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "references" / "context_map.md",
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "scripts" / "nbv_context_index.sh",
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "scripts" / "nbv_literature_index.sh",
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "scripts" / "nbv_literature_search.sh",
    ROOT_DIR / "scripts" / "validate_mermaid.sh",
]

REQUIRED_FRONTMATTER_KEYS = [
    "id",
    "date",
    "title",
    "status",
    "topics",
    "confidence",
    "canonical_updates_needed",
]

PATH_TOKEN_RE = re.compile(
    r"(?P<path>"
    r"(?:\.agents|docs|aria_nbv|scripts|archive)/[A-Za-z0-9_./-]+"
    r"|/home/jandu/repos/NBV/[A-Za-z0-9_./-]+"
    r")"
)

FRONTMATTER_KEY_RE = r"^{key}\s*:"
LIST_ITEM_RE = re.compile(r"^\s*-\s+(?P<value>.+?)\s*$", re.MULTILINE)
PLACEHOLDER_PATH_SEGMENTS = {"YYYY", "MM", "DD"}

STALE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\boracle_rri/"), "stale workspace path `oracle_rri/`"),
    (re.compile(r"\boracle_rri\.(?!py\b)[A-Za-z_]"), "stale Python import path `oracle_rri.*`"),
    (re.compile(r"\$\{ROOT_DIR\}/literature\b"), "stale literature root `${ROOT_DIR}/literature`"),
    (re.compile(r"(?<!docs/)literature/tex-src"), "stale literature path `literature/tex-src`"),
]

REMOTE_FETCH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnpx\s+-y\b"), "runtime package fetch `npx -y`"),
    (re.compile(r"\bcurl(?:\s|$)"), "runtime remote fetch `curl`"),
    (re.compile(r"\bwget(?:\s|$)"), "runtime remote fetch `wget`"),
    (re.compile(r"https?://"), "raw remote URL"),
]

SCAFFOLD_DEBRIS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?m)<!--\s*TODO\b|^\s*TODO\b"), "unresolved scaffold marker `TODO`"),
    (re.compile(r"(?m)<!--\s*FIXME\b|^\s*FIXME\b"), "unresolved scaffold marker `FIXME`"),
    (re.compile(r"(?m)<!--\s*XXX\b|^\s*XXX\b"), "unresolved scaffold marker `XXX`"),
    (re.compile(r"(?m)^\s*-\s*$"), "empty placeholder bullet"),
]

REQUIRED_SUBSTRINGS: dict[Path, tuple[str, ...]] = {
    ROOT_DIR / "AGENTS.md": (".agents/memory/state/OWNER_DIRECTIVES.md",),
    ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "SKILL.md": (".agents/memory/state/OWNER_DIRECTIVES.md",),
}

REQUIRED_AGENTS_FRONTMATTER_KEYS = ("scope", "applies_to", "summary")
REPO_RELATIVE_PATH_POLICY_FILES = (
    ROOT_DIR / ".agents" / "memory" / "state" / "DECISIONS.md",
    ROOT_DIR / ".agents" / "memory" / "state" / "OWNER_DIRECTIVES.md",
)


@dataclass(frozen=True)
class ValidationError:
    path: Path
    message: str

    def render(self) -> str:
        try:
            rel = self.path.relative_to(ROOT_DIR)
        except ValueError:
            rel = self.path
        return f"{rel}: {self.message}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def discover_agents_files() -> tuple[Path, ...]:
    paths: list[Path] = [ROOT_AGENTS_FILE]
    for root in AGENTS_SEARCH_ROOTS:
        if root.exists():
            paths.extend(sorted(root.rglob("AGENTS.md")))
    return tuple(dict.fromkeys(paths))


def is_boundary_guide(path: Path) -> bool:
    return path.name == "AGENTS.md" and any(path.is_relative_to(root) for root in BOUNDARY_GUIDE_ROOTS)


def split_frontmatter(path: Path, text: str) -> tuple[str, str] | ValidationError:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ValidationError(path, "missing YAML frontmatter")

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            frontmatter = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1 :])
            return frontmatter, body
    return ValidationError(path, "frontmatter is missing a closing `---`")


def has_frontmatter_key(frontmatter: str, key: str) -> bool:
    return re.search(FRONTMATTER_KEY_RE.format(key=re.escape(key)), frontmatter, re.MULTILINE) is not None


def extract_list_values(frontmatter: str, key: str) -> list[str]:
    match = re.search(
        rf"^{re.escape(key)}\s*:\s*(?P<inline>\[[^\n]*\])?\s*(?:\n(?P<body>(?:[ \t]+-\s+.*(?:\n|$))+))?",
        frontmatter,
        re.MULTILINE,
    )
    if match is None:
        return []

    values: list[str] = []
    inline = match.group("inline")
    if inline:
        stripped = inline.strip()[1:-1].strip()
        if not stripped:
            return []
        for value in stripped.split(","):
            values.append(value.strip().strip("\"'"))
        return values

    body = match.group("body")
    if not body:
        return []
    for item in LIST_ITEM_RE.finditer(body):
        values.append(item.group("value").strip().strip("\"'"))
    return values


def collect_path_tokens(text: str) -> set[str]:
    return {match.group("path").rstrip(".,:`") for match in PATH_TOKEN_RE.finditer(text)}


def normalize_repo_path(raw: str) -> Path:
    if raw.startswith("/home/jandu/repos/NBV/"):
        return Path(raw)
    return ROOT_DIR / raw


def is_placeholder_path(raw: str) -> bool:
    return any(segment in PLACEHOLDER_PATH_SEGMENTS for segment in Path(raw).parts)


def validate_memory_entry(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    split = split_frontmatter(path, text)
    if isinstance(split, ValidationError):
        return [split]

    frontmatter, body = split
    status_match = re.search(r"^status\s*:\s*(?P<value>.+?)\s*$", frontmatter, re.MULTILINE)
    status = status_match.group("value").strip().strip("\"'") if status_match else ""
    if status == "legacy-imported":
        return []

    for key in REQUIRED_FRONTMATTER_KEYS:
        if not has_frontmatter_key(frontmatter, key):
            errors.append(ValidationError(path, f"missing required frontmatter key `{key}`"))

    if "## Prompt Follow-Through" not in body:
        errors.append(ValidationError(path, "missing required `## Prompt Follow-Through` section"))

    for update_path in extract_list_values(frontmatter, "canonical_updates_needed"):
        candidate = normalize_repo_path(update_path)
        if not candidate.exists():
            errors.append(
                ValidationError(
                    path,
                    f"`canonical_updates_needed` references missing path `{update_path}`",
                )
            )
    return errors


def validate_memory_history() -> list[ValidationError]:
    errors: list[ValidationError] = []
    for path in sorted(HISTORY_DIR.rglob("*.md")):
        text = read_text(path)
        errors.extend(validate_memory_entry(path, text))
    return errors


def validate_stale_patterns(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for pattern, description in STALE_PATTERNS:
        if pattern.search(text):
            errors.append(ValidationError(path, f"contains {description}"))
    return errors


def validate_path_references(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for raw_path in sorted(collect_path_tokens(text)):
        if is_placeholder_path(raw_path):
            continue
        candidate = normalize_repo_path(raw_path)
        if not candidate.exists():
            errors.append(ValidationError(path, f"references missing path `{raw_path}`"))
    return errors


def validate_remote_fetch_patterns(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for pattern, description in REMOTE_FETCH_PATTERNS:
        if pattern.search(text):
            errors.append(
                ValidationError(
                    path,
                    f"contains disallowed {description}; use a repo wrapper or preinstalled local tool",
                )
            )
    return errors


def validate_required_headings(path: Path, text: str, agents_files: set[Path]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    headings: list[str] = []
    if path in agents_files and path != ROOT_AGENTS_FILE:
        headings.extend(("## Verification", "## Completion Criteria"))
    if path in agents_files and is_boundary_guide(path):
        headings.extend(("## Public Contracts", "## Boundary Rules"))
    for heading in headings:
        if heading not in text:
            errors.append(ValidationError(path, f"missing required heading `{heading}`"))
    return errors


def validate_required_substrings(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for needle in REQUIRED_SUBSTRINGS.get(path, ()):
        if needle not in text:
            errors.append(ValidationError(path, f"missing required scaffold reference `{needle}`"))
    return errors


def validate_scaffold_debris(path: Path, text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for pattern, description in SCAFFOLD_DEBRIS_PATTERNS:
        if pattern.search(text):
            errors.append(ValidationError(path, f"contains {description}"))
    return errors


def validate_repo_relative_paths(path: Path, text: str) -> list[ValidationError]:
    if "/home/jandu/repos/NBV/" in text:
        return [ValidationError(path, "contains absolute workspace paths; use repo-relative paths in scaffold guidance")]
    return []


def validate_agents_frontmatter(path: Path, text: str) -> list[ValidationError]:
    split = split_frontmatter(path, text)
    if isinstance(split, ValidationError):
        return [split]

    frontmatter, _body = split
    errors: list[ValidationError] = []
    for key in REQUIRED_AGENTS_FRONTMATTER_KEYS:
        if not has_frontmatter_key(frontmatter, key):
            errors.append(ValidationError(path, f"missing required AGENTS frontmatter key `{key}`"))
    return errors


def validate_source_index() -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not SOURCE_INDEX.exists():
        return [ValidationError(SOURCE_INDEX, "missing generated source index; run `make context`")]

    text = read_text(SOURCE_INDEX)
    actual_families = sum(1 for child in LITERATURE_TEX_SRC.iterdir() if child.is_dir()) if LITERATURE_TEX_SRC.exists() else 0
    if actual_families > 0 and "| Literature | 0 families" in text:
        errors.append(
            ValidationError(
                SOURCE_INDEX,
                "reports zero literature families even though docs/literature/tex-src contains sources",
            )
        )
    if 'rg -n "<term>" literature/' in text:
        errors.append(ValidationError(SOURCE_INDEX, "uses stale root-level literature search recipe"))
    if ".agents/memory/state/PROJECT_STATE.md" not in text:
        errors.append(ValidationError(SOURCE_INDEX, "does not advertise PROJECT_STATE.md in the retrieval ladder"))
    if ".agents/memory/state/OWNER_DIRECTIVES.md" not in text:
        errors.append(ValidationError(SOURCE_INDEX, "does not advertise OWNER_DIRECTIVES.md in the retrieval ladder"))
    if "## Path-local boundary guides" not in text:
        errors.append(ValidationError(SOURCE_INDEX, "does not advertise the path-local boundary-guides section"))
    for guide in SOURCE_INDEX_PATH_LOCAL_GUIDES:
        guide_str = guide.relative_to(ROOT_DIR).as_posix()
        if guide_str not in text:
            errors.append(ValidationError(SOURCE_INDEX, f"does not advertise boundary guide `{guide_str}`"))
    return errors


def validate_scaffold() -> list[ValidationError]:
    errors: list[ValidationError] = []
    agents_files = discover_agents_files()
    agents_set = set(agents_files)
    scaffold_files = [*agents_files, *NON_AGENT_SCAFFOLD_FILES]
    for path in scaffold_files:
        if not path.exists():
            errors.append(ValidationError(path, "expected scaffold file is missing"))
            continue
        text = read_text(path)
        if path in agents_set:
            errors.extend(validate_agents_frontmatter(path, text))
        if path in agents_set or path in REPO_RELATIVE_PATH_POLICY_FILES:
            errors.extend(validate_repo_relative_paths(path, text))
        errors.extend(validate_stale_patterns(path, text))
        errors.extend(validate_remote_fetch_patterns(path, text))
        errors.extend(validate_path_references(path, text))
        errors.extend(validate_required_headings(path, text, agents_set))
        errors.extend(validate_required_substrings(path, text))
        errors.extend(validate_scaffold_debris(path, text))
    errors.extend(validate_source_index())
    return errors


def run_self_test() -> list[str]:
    sample = (
        "Use `oracle_rri/.venv/bin/python`, "
        "`literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex`, and "
        "`npx -y @mermaid-js/mermaid-cli -i in.mmd -o out.svg`."
    )
    findings = [description for pattern, description in STALE_PATTERNS if pattern.search(sample)]
    if "stale workspace path `oracle_rri/`" not in findings:
        return ["self-test failed to detect stale workspace paths"]
    if "stale literature path `literature/tex-src`" not in findings:
        return ["self-test failed to detect stale literature paths"]
    remote_findings = [description for pattern, description in REMOTE_FETCH_PATTERNS if pattern.search(sample)]
    if "runtime package fetch `npx -y`" not in remote_findings:
        return ["self-test failed to detect disallowed runtime package fetches"]
    sample_agents = {
        ROOT_AGENTS_FILE,
        ROOT_DIR / "docs" / "AGENTS.md",
        ROOT_DIR / "aria_nbv" / "aria_nbv" / "vin" / "AGENTS.md",
    }
    heading_errors = validate_required_headings(ROOT_DIR / "docs" / "AGENTS.md", "## Verification\n", sample_agents)
    if not any("## Completion Criteria" in error.message for error in heading_errors):
        return ["self-test failed to detect missing completion-criteria headings"]
    boundary_heading_errors = validate_required_headings(
        ROOT_DIR / "aria_nbv" / "aria_nbv" / "vin" / "AGENTS.md",
        "## Verification\n## Completion Criteria\n",
        sample_agents,
    )
    if not any("## Public Contracts" in error.message for error in boundary_heading_errors):
        return ["self-test failed to detect missing boundary-guide headings"]
    debris_errors = validate_scaffold_debris(ROOT_DIR / "AGENTS.md", "<!-- TODO: clean me -->\n")
    if not any("TODO" in error.message for error in debris_errors):
        return ["self-test failed to detect unresolved scaffold markers"]
    frontmatter_errors = validate_agents_frontmatter(ROOT_AGENTS_FILE, "# no frontmatter\n")
    if not any("missing YAML frontmatter" in error.message for error in frontmatter_errors):
        return ["self-test failed to detect missing AGENTS frontmatter"]
    relative_path_errors = validate_repo_relative_paths(ROOT_AGENTS_FILE, "/home/jandu/repos/NBV/AGENTS.md")
    if not any("repo-relative paths" in error.message for error in relative_path_errors):
        return ["self-test failed to detect absolute workspace paths in scaffold guidance"]
    history_errors = validate_memory_entry(
        ROOT_DIR / ".agents" / "memory" / "history" / "sample.md",
        "---\n"
        "id: sample\n"
        "date: 2026-03-29\n"
        "title: Sample\n"
        "status: done\n"
        "topics: [scaffold]\n"
        "confidence: high\n"
        "canonical_updates_needed: []\n"
        "---\n\n"
        "# Task\n\n"
        "sample\n",
    )
    if not any("Prompt Follow-Through" in error.message for error in history_errors):
        return ["self-test failed to detect missing prompt follow-through sections"]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-only", action="store_true", help="validate debrief hygiene only")
    parser.add_argument("--scaffold-only", action="store_true", help="validate scaffold routing only")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a dry validator self-test that proves stale-path checks would fire",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.memory_only and args.scaffold_only:
        print("cannot combine --memory-only and --scaffold-only", file=sys.stderr)
        return 2

    self_test_errors = run_self_test() if args.self_test else []
    if self_test_errors:
        for error in self_test_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        print("Validator self-test passed.")
        if not args.memory_only and not args.scaffold_only:
            return 0

    errors: list[ValidationError] = []
    if not args.scaffold_only:
        errors.extend(validate_memory_history())
    if not args.memory_only:
        errors.extend(validate_scaffold())

    if errors:
        for error in errors:
            print(f"ERROR: {error.render()}", file=sys.stderr)
        return 1

    modes: list[str] = []
    if not args.scaffold_only:
        modes.append("memory")
    if not args.memory_only:
        modes.append("scaffold")
    print(f"Validated {' + '.join(modes)} surfaces successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

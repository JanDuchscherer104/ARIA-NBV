"""Autoimprove helpers for simplifying the Aria-NBV Python workspace.

The implementation adapts the `karpathy/autoresearch` pattern to a codebase
that has many editable modules instead of a single `train.py`. A single
Markdown file with YAML front matter defines the editable scope, canonical
owners, helper-policy roots, scoring weights, and mode-specific verification
commands. This module parses that specification, audits the repository for
duplicate contracts and helper sprawl, renders a mode-specific prompt, and
computes a score for the current repo state.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AdjacentModuleGroup:
    """Pair of adjacent module roots that should converge to one owner.

    Attributes:
        name: Stable identifier for the adjacency group.
        left: Repo-relative path to the left module root.
        right: Repo-relative path to the right module root.
        min_similarity: Minimum normalized similarity for a duplicate pair hit.
        canonical_owner: Optional repo-relative root that should own duplicates.
    """

    name: str
    """Stable identifier for the adjacency group."""

    left: str
    """Repo-relative path to the left module root."""

    right: str
    """Repo-relative path to the right module root."""

    min_similarity: float = 0.8
    """Minimum normalized similarity for a duplicate pair hit."""

    canonical_owner: str | None = None
    """Optional repo-relative root that should own duplicates in this group."""


@dataclass(slots=True)
class CanonicalOwnerRule:
    """Heuristic rule for selecting one canonical owner for duplicate symbols.

    Attributes:
        name: Stable identifier for the rule.
        matches: Path prefixes that must all be present among the locations.
        prefer: Repo-relative path prefix that should own the symbol.
    """

    name: str
    """Stable identifier for the rule."""

    matches: list[str]
    """Path prefixes that must all be present among the duplicate locations."""

    prefer: str
    """Repo-relative path prefix that should own the symbol."""


@dataclass(slots=True)
class ImportPreferenceRule:
    """Preference rule that flags imports through a legacy compatibility surface.

    Attributes:
        name: Stable identifier for the rule.
        legacy_prefix: Imported module prefix that should be phased out.
        canonical_prefix: Canonical module prefix that should replace it.
        allow_paths: Repo-relative paths that may still import through the legacy surface.
    """

    name: str
    """Stable identifier for the rule."""

    legacy_prefix: str
    """Imported module prefix that should be phased out."""

    canonical_prefix: str
    """Canonical module prefix that should replace it."""

    allow_paths: list[str] = field(default_factory=list)
    """Repo-relative paths that may still import through the legacy surface."""


@dataclass(slots=True)
class CostFunction:
    """Machine-readable scoring definition loaded from the Markdown spec.

    Attributes:
        objective: Optimization direction, usually ``"maximize"``.
        expression: Human-readable score expression.
        weights: Feature weights used by ``compute_score``.
    """

    objective: str
    """Optimization direction, usually ``"maximize"``."""

    expression: str
    """Human-readable score expression."""

    weights: dict[str, float]
    """Feature weights used by ``compute_score``."""


@dataclass(slots=True)
class ModeSpec:
    """Mode-specific goal, focus areas, and verification commands.

    Attributes:
        goal: Short goal statement for the mode.
        focus: Bulleted focus areas for the mode.
        verify_commands: Shell commands that validate the mode's changes.
    """

    goal: str
    """Short goal statement for the mode."""

    focus: list[str] = field(default_factory=list)
    """Bulleted focus areas for the mode."""

    verify_commands: list[str] = field(default_factory=list)
    """Shell commands that validate the mode's changes."""


@dataclass(slots=True)
class AutoImproveSpec:
    """Resolved autoimprove spec with repo-relative paths.

    Attributes:
        path: Absolute path to the loaded Markdown spec.
        repo_root: Absolute repo root resolved from the spec path.
        name: Human-readable name of the autoimprove program.
        executor: Human-facing name of the code-change executor.
        report_dir: Repo-relative directory used for generated reports.
        diff_base: Git base ref used for diff metrics.
        audit_paths: Repo-relative paths included in redundancy audits.
        editable_paths: Repo-relative paths allowed to change.
        protected_paths: Repo-relative paths that should stay untouched.
        adjacent_module_groups: Adjacent module roots to compare for duplicates.
        canonical_owner_rules: Heuristics for choosing one canonical owner.
        shared_helper_roots: Repo-relative roots where shared helpers may live.
        ignored_helper_names: Top-level helper names ignored by the audit.
        helper_density_threshold: Minimum private helper count before a module is flagged.
        cost_function: Machine-readable scoring definition.
        modes: Available autoimprove modes.
        default_mode: Default mode to render when none is provided.
        prompt_body: Markdown body below the YAML front matter.
    """

    path: Path
    """Absolute path to the loaded Markdown spec."""

    repo_root: Path
    """Absolute repo root resolved from the spec path."""

    name: str
    """Human-readable name of the autoimprove program."""

    executor: str
    """Human-facing name of the code-change executor."""

    report_dir: str
    """Repo-relative directory used for generated reports."""

    diff_base: str
    """Git base ref used for diff metrics."""

    audit_paths: list[str]
    """Repo-relative paths included in redundancy audits."""

    editable_paths: list[str]
    """Repo-relative paths allowed to change."""

    protected_paths: list[str]
    """Repo-relative paths that should stay untouched."""

    adjacent_module_groups: list[AdjacentModuleGroup]
    """Adjacent module roots to compare for duplicates."""

    canonical_owner_rules: list[CanonicalOwnerRule]
    """Heuristics for choosing one canonical owner."""

    import_preference_rules: list[ImportPreferenceRule]
    """Rules that flag imports routed through legacy compatibility surfaces."""

    shared_helper_roots: list[str]
    """Repo-relative roots where shared helpers may live."""

    ignored_helper_names: set[str]
    """Top-level helper names ignored by the audit."""

    helper_density_threshold: int
    """Minimum private helper count before a module is flagged."""

    cost_function: CostFunction
    """Machine-readable scoring definition."""

    modes: dict[str, ModeSpec]
    """Available autoimprove modes."""

    default_mode: str
    """Default mode to render when none is provided."""

    prompt_body: str
    """Markdown body below the YAML front matter."""


@dataclass(slots=True)
class DuplicateModulePair:
    """Near-duplicate file pair found in an adjacent module group.

    Attributes:
        group: Adjacency group that produced the match.
        left: Repo-relative path to the left file.
        right: Repo-relative path to the right file.
        similarity: Normalized source similarity in ``[0, 1]``.
        suggested_owner: Repo-relative path prefix that should own the pair.
    """

    group: str
    """Adjacency group that produced the match."""

    left: str
    """Repo-relative path to the left file."""

    right: str
    """Repo-relative path to the right file."""

    similarity: float
    """Normalized source similarity in ``[0, 1]``."""

    suggested_owner: str | None = None
    """Repo-relative path prefix that should own the pair."""


@dataclass(slots=True)
class RepeatedSymbolGroup:
    """Repeated symbol definition that should collapse to one owner.

    Attributes:
        name: Symbol name.
        kind: Contract or helper category.
        locations: Repo-relative file paths where the symbol is defined.
        suggested_owner: Repo-relative path or prefix that should own it.
    """

    name: str
    """Symbol name."""

    kind: str
    """Contract or helper category."""

    locations: list[str]
    """Repo-relative file paths where the symbol is defined."""

    suggested_owner: str | None = None
    """Repo-relative path or prefix that should own the symbol."""


@dataclass(slots=True)
class PrivateHelperModule:
    """Private top-level helper functions defined outside shared helper roots.

    Attributes:
        path: Repo-relative Python file path.
        helpers: Helper function names defined in the file.
        suggested_owner: Suggested shared helper root or owner module.
    """

    path: str
    """Repo-relative Python file path."""

    helpers: list[str]
    """Helper function names defined in the file."""

    suggested_owner: str | None = None
    """Suggested shared helper root or owner module."""


@dataclass(slots=True)
class PrivateExportModule:
    """Module that exports underscore-prefixed names in `__all__`.

    Attributes:
        path: Repo-relative Python file path.
        names: Exported underscore-prefixed names.
    """

    path: str
    """Repo-relative Python file path."""

    names: list[str]
    """Exported underscore-prefixed names."""


@dataclass(slots=True)
class LegacyImportEdge:
    """Import edge that still routes through a configured legacy module prefix.

    Attributes:
        importer: Repo-relative file that performs the import.
        imported_module: Resolved imported module path.
        names: Imported symbol names for the edge.
        canonical_prefix: Canonical module prefix that should replace the legacy prefix.
    """

    importer: str
    """Repo-relative file that performs the import."""

    imported_module: str
    """Resolved imported module path."""

    names: list[str]
    """Imported symbol names for the edge."""

    canonical_prefix: str
    """Canonical module prefix that should replace the legacy prefix."""


@dataclass(slots=True)
class AuditReport:
    """Current redundancy state of the repo.

    Attributes:
        python_loc: Non-empty, non-comment Python LOC inside editable scope.
        duplicate_module_pairs: Near-duplicate adjacent module pairs.
        repeated_class_groups: Repeated contract/model definitions.
        helper_collisions: Repeated top-level helper definitions.
        private_helper_modules: Private helpers defined outside shared roots.
        private_export_modules: Modules exporting underscore-prefixed names.
        legacy_import_edges: Imports still routed through a legacy compatibility prefix.
    """

    python_loc: int
    """Non-empty, non-comment Python LOC inside editable scope."""

    duplicate_module_pairs: list[DuplicateModulePair]
    """Near-duplicate adjacent module pairs."""

    repeated_class_groups: list[RepeatedSymbolGroup]
    """Repeated contract/model definitions."""

    helper_collisions: list[RepeatedSymbolGroup]
    """Repeated top-level helper definitions."""

    private_helper_modules: list[PrivateHelperModule]
    """Private helpers defined outside shared roots."""

    private_export_modules: list[PrivateExportModule]
    """Modules exporting underscore-prefixed names."""

    legacy_import_edges: list[LegacyImportEdge]
    """Imports still routed through a legacy compatibility prefix."""


@dataclass(slots=True)
class DiffMetrics:
    """Git-based metrics used by the score function.

    Attributes:
        additions: Added Python lines relative to the configured base.
        deletions: Deleted Python lines relative to the configured base.
        net_python_lines_removed: ``deletions - additions``.
        protected_path_touches: Number of touched protected paths.
    """

    additions: int
    """Added Python lines relative to the configured base."""

    deletions: int
    """Deleted Python lines relative to the configured base."""

    net_python_lines_removed: int
    """``deletions - additions``."""

    protected_path_touches: int
    """Number of touched protected paths."""


@dataclass(slots=True)
class VerificationCommandResult:
    """Outcome of one configured verification command.

    Attributes:
        command: Shell command that ran.
        returncode: Process return code.
    """

    command: str
    """Shell command that ran."""

    returncode: int
    """Process return code."""


@dataclass(slots=True)
class VerificationSummary:
    """Aggregate verification state for the active mode.

    Attributes:
        pass_rate: Fraction of successful verification commands.
        results: Per-command outcomes.
    """

    pass_rate: float
    """Fraction of successful verification commands."""

    results: list[VerificationCommandResult]
    """Per-command outcomes."""


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split Markdown into YAML front matter and body text.

    Args:
        text: Raw Markdown file contents.

    Returns:
        Tuple of parsed YAML mapping and Markdown body.

    Raises:
        ValueError: If the file lacks valid YAML front matter.
    """

    if not text.startswith("---\n"):
        raise ValueError("Expected Markdown file with YAML front matter.")
    _, _, rest = text.partition("---\n")
    header, sep, body = rest.partition("\n---\n")
    if not sep:
        raise ValueError("Could not find closing YAML front matter fence.")
    payload = yaml.safe_load(header) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML front matter must deserialize to a mapping.")
    return payload, body.strip()


def _default_spec_path(start: Path | None = None) -> Path:
    """Resolve the nearest ``autoimprove.md`` at or above the current cwd.

    Args:
        start: Optional starting directory. Defaults to the current cwd.

    Returns:
        Path to the nearest ``autoimprove.md``.

    Raises:
        FileNotFoundError: If no spec is found in the current directory tree.
    """

    current = (start or Path.cwd()).resolve()
    for candidate_root in [current, *current.parents]:
        candidate = candidate_root / "autoimprove.md"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate autoimprove.md in the current directory tree.")


def load_autoimprove_spec(path: str | Path | None = None) -> AutoImproveSpec:
    """Load the repo's autoimprove spec from Markdown front matter.

    Args:
        path: Optional path to the Markdown spec.

    Returns:
        Resolved autoimprove specification.
    """

    spec_path = Path(path) if path is not None else _default_spec_path()
    spec_path = spec_path.expanduser().resolve()
    payload, body = _split_front_matter(spec_path.read_text())
    repo_root = spec_path.parent

    modes = {
        name: ModeSpec(
            goal=str(mode_payload["goal"]),
            focus=[str(item) for item in mode_payload.get("focus", [])],
            verify_commands=[str(item) for item in mode_payload.get("verify_commands", [])],
        )
        for name, mode_payload in payload["modes"].items()
    }

    return AutoImproveSpec(
        path=spec_path,
        repo_root=repo_root,
        name=str(payload["name"]),
        executor=str(payload.get("executor", "codex")),
        report_dir=str(payload.get("report_dir", ".agents/workspace/autoimprove/reports")),
        diff_base=str(payload.get("diff_base", "origin/main")),
        audit_paths=[str(item) for item in payload.get("audit_paths", payload.get("editable_paths", []))],
        editable_paths=[str(item) for item in payload.get("editable_paths", [])],
        protected_paths=[str(item) for item in payload.get("protected_paths", [])],
        adjacent_module_groups=[
            AdjacentModuleGroup(
                name=str(item["name"]),
                left=str(item["left"]),
                right=str(item["right"]),
                min_similarity=float(item.get("min_similarity", 0.8)),
                canonical_owner=str(item["canonical_owner"]) if item.get("canonical_owner") is not None else None,
            )
            for item in payload.get("adjacent_module_groups", [])
        ],
        canonical_owner_rules=[
            CanonicalOwnerRule(
                name=str(item["name"]),
                matches=[str(match) for match in item.get("matches", [])],
                prefer=str(item["prefer"]),
            )
            for item in payload.get("canonical_owner_rules", [])
        ],
        import_preference_rules=[
            ImportPreferenceRule(
                name=str(item["name"]),
                legacy_prefix=str(item["legacy_prefix"]),
                canonical_prefix=str(item["canonical_prefix"]),
                allow_paths=[str(match) for match in item.get("allow_paths", [])],
            )
            for item in payload.get("import_preference_rules", [])
        ],
        shared_helper_roots=[str(item) for item in payload.get("shared_helper_roots", [])],
        ignored_helper_names={str(item) for item in payload.get("ignored_helper_names", [])},
        helper_density_threshold=int(payload.get("helper_density_threshold", 1)),
        cost_function=CostFunction(
            objective=str(payload["cost_function"]["objective"]),
            expression=str(payload["cost_function"]["expression"]),
            weights={str(key): float(value) for key, value in payload["cost_function"]["weights"].items()},
        ),
        modes=modes,
        default_mode=str(payload["default_mode"]),
        prompt_body=body,
    )


def _resolve_paths(repo_root: Path, paths: list[str]) -> list[Path]:
    """Resolve spec-relative paths against the repo root.

    Args:
        repo_root: Absolute repo root.
        paths: Repo-relative paths.

    Returns:
        Resolved absolute paths.
    """

    return [(repo_root / path).resolve() for path in paths]


def _iter_python_files(paths: list[Path]) -> list[Path]:
    """Collect Python files under a list of files or directories.

    Args:
        paths: Absolute files or directories.

    Returns:
        Sorted list of Python source files.
    """

    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
            continue
        if not path.exists():
            continue
        files.extend(sorted(path.rglob("*.py")))
    return [path for path in files if "__pycache__" not in path.parts]


def _normalized_source(path: Path) -> str:
    """Normalize a Python file for lightweight similarity comparisons.

    Args:
        path: Python file path.

    Returns:
        Source text without blank lines or comments.
    """

    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _relative_to(path: Path, root: Path) -> str:
    """Render a stable repo-relative path string.

    Args:
        path: Absolute path to render.
        root: Absolute repo root.

    Returns:
        Repo-relative POSIX path.
    """

    return path.resolve().relative_to(root.resolve()).as_posix()


def _count_python_loc(paths: list[Path]) -> int:
    """Count non-empty, non-comment Python lines.

    Args:
        paths: Python file paths.

    Returns:
        Total non-empty, non-comment Python LOC.
    """

    total = 0
    for path in paths:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def _decorator_name(node: ast.expr) -> str | None:
    """Return the normalized decorator name for one AST decorator.

    Args:
        node: Decorator AST node.

    Returns:
        Normalized decorator name, if recognizable.
    """

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _base_name(node: ast.expr) -> str | None:
    """Return the normalized base-class name for one AST expression.

    Args:
        node: Base-class AST node.

    Returns:
        Normalized base name, if recognizable.
    """

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _class_contract_kind(node: ast.ClassDef) -> str | None:
    """Classify a top-level class as a type/model contract of interest.

    Args:
        node: Top-level ``ClassDef`` node.

    Returns:
        Contract kind string or ``None`` if the class is not audit-relevant.
    """

    decorators = {_decorator_name(item) for item in node.decorator_list}
    base_names = {_base_name(item) for item in node.bases}
    suffix_map = {
        "batch": ("Batch",),
        "config": ("Config",),
        "contract": ("Entry", "Manifest", "Metadata", "Result", "Sample", "Spec", "View"),
        "runtime_surface": ("Dataset", "Provider", "Writer"),
    }

    if "dataclass" in decorators:
        for kind, suffixes in suffix_map.items():
            if node.name.endswith(suffixes):
                return kind
        return "dataclass"
    if "BaseConfig" in base_names or node.name.endswith("Config"):
        return "config"
    if "BaseModel" in base_names or "BaseSettings" in base_names:
        return "pydantic_model"
    if "Protocol" in base_names:
        return "protocol"
    if "TypedDict" in base_names:
        return "typed_dict"
    if "Struct" in base_names:
        return "msgspec_struct"
    for kind, suffixes in suffix_map.items():
        if node.name.endswith(suffixes):
            return kind
    return None


def _typed_dict_assignment_kind(node: ast.Assign) -> str | None:
    """Detect top-level ``TypedDict`` alias assignments.

    Args:
        node: Top-level ``Assign`` node.

    Returns:
        ``"typed_dict"`` when the assignment defines a ``TypedDict`` alias.
    """

    if not isinstance(node.value, ast.Call):
        return None
    func_name = _base_name(node.value.func)
    return "typed_dict" if func_name == "TypedDict" else None


def _location_matches(path: str, pattern: str) -> bool:
    """Check whether one repo-relative path matches a configured prefix.

    Args:
        path: Repo-relative file path.
        pattern: Repo-relative prefix or exact file path.

    Returns:
        Whether the path matches the pattern.
    """

    normalized = pattern.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def _best_matching_location(locations: list[str], preferred_prefix: str) -> str | None:
    """Return the best concrete location under one preferred prefix.

    Args:
        locations: Repo-relative duplicate locations.
        preferred_prefix: Repo-relative owner prefix.

    Returns:
        Best matching concrete location, if any.
    """

    matches = [location for location in locations if _location_matches(location, preferred_prefix)]
    if not matches:
        return None
    matches.sort(key=lambda item: ("/experimental/" in item, item.endswith("_old.py"), len(item), item))
    return matches[0]


def _suggest_owner(locations: list[str], spec: AutoImproveSpec) -> str | None:
    """Suggest a canonical owner for a set of duplicate locations.

    Args:
        locations: Repo-relative duplicate locations.
        spec: Loaded autoimprove specification.

    Returns:
        Suggested canonical owner path or prefix.
    """

    for rule in spec.canonical_owner_rules:
        if rule.matches and all(
            any(_location_matches(location, pattern) for location in locations) for pattern in rule.matches
        ):
            preferred_location = _best_matching_location(locations, rule.prefer)
            return preferred_location or rule.prefer

    data_handling_location = _best_matching_location(locations, "aria_nbv/aria_nbv/data_handling")
    if data_handling_location is not None and any(
        _location_matches(location, "aria_nbv/aria_nbv/data") for location in locations
    ):
        return data_handling_location

    main_vin_location = _best_matching_location(locations, "aria_nbv/aria_nbv/vin")
    if main_vin_location is not None and any("/vin/experimental/" in location for location in locations):
        return main_vin_location

    non_old_locations = [location for location in locations if not location.endswith("_old.py")]
    if non_old_locations and len(non_old_locations) != len(locations):
        return sorted(non_old_locations, key=lambda item: ("/experimental/" in item, len(item), item))[0]

    return sorted(locations, key=lambda item: ("/experimental/" in item, item.endswith("_old.py"), len(item), item))[0]


def _find_adjacent_group_owner(spec: AutoImproveSpec, group_name: str) -> str | None:
    """Resolve the configured owner prefix for one adjacency group.

    Args:
        spec: Loaded autoimprove specification.
        group_name: Adjacency group identifier.

    Returns:
        Repo-relative owner prefix, if configured.
    """

    for group in spec.adjacent_module_groups:
        if group.name == group_name:
            return group.canonical_owner
    return None


def _is_under_shared_helper_root(path: str, spec: AutoImproveSpec) -> bool:
    """Check whether a repo-relative file is inside a shared helper root.

    Args:
        path: Repo-relative file path.
        spec: Loaded autoimprove specification.

    Returns:
        Whether the file is under a configured shared helper root.
    """

    return any(_location_matches(path, root) for root in spec.shared_helper_roots)


def _iter_top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Collect top-level function definitions from one module AST.

    Args:
        tree: Parsed module AST.

    Returns:
        Top-level function definitions.
    """

    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _module_name_for_path(relative_path: str) -> tuple[str, bool]:
    """Resolve a repo-relative Python file path to a best-effort module name.

    Args:
        relative_path: Repo-relative Python file path.

    Returns:
        Tuple of resolved dotted module name and whether the file is `__init__.py`.
    """

    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    is_package_init = parts[-1] == "__init__"
    if is_package_init:
        parts = parts[:-1]
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = parts[1:]
    return ".".join(parts), is_package_init


def _resolve_import_module(relative_path: str, node: ast.ImportFrom) -> str:
    """Resolve one `ImportFrom` node to a dotted module path.

    Args:
        relative_path: Repo-relative path of the importing file.
        node: AST `ImportFrom` node.

    Returns:
        Resolved imported module path, or an empty string if it cannot be resolved.
    """

    if node.level == 0:
        return node.module or ""

    current_module, is_package_init = _module_name_for_path(relative_path)
    package_parts = current_module.split(".") if current_module else []
    if not is_package_init and package_parts:
        package_parts = package_parts[:-1]
    levels_up = max(node.level - 1, 0)
    if levels_up > len(package_parts):
        return node.module or ""
    base_parts = package_parts[: len(package_parts) - levels_up]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(part for part in base_parts if part)


def _extract_private_exports(tree: ast.Module, *, ignored_names: set[str]) -> list[str]:
    """Collect underscore-prefixed names exported through module `__all__`.

    Args:
        tree: Parsed module AST.
        ignored_names: Names the audit should ignore.

    Returns:
        Sorted exported underscore-prefixed names.
    """

    names: set[str] = set()
    for node in tree.body:
        values: list[ast.expr] | None = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                    values = list(node.value.elts)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                    values = list(node.value.elts)
        if values is None:
            continue
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                name = value.value
                if name.startswith("_") and name not in ignored_names:
                    names.add(name)
    return sorted(names)


def _match_adjacent_module_pairs(
    spec: AutoImproveSpec,
    group: AdjacentModuleGroup,
) -> list[DuplicateModulePair]:
    """Find the best high-similarity matches across two adjacent module roots.

    Same-basename matching misses the important refactor seams where a module
    was renamed while its implementation stayed almost identical, such as
    ``offline_cache.py`` versus ``oracle_cache.py``. This matcher considers all
    left/right pairs and greedily keeps the highest-similarity non-overlapping
    matches so those renamed duplicates remain visible in the audit.

    Args:
        spec: Loaded autoimprove specification.
        group: One configured adjacent module group.

    Returns:
        High-similarity duplicate pairs for the group.
    """

    left_root = (spec.repo_root / group.left).resolve()
    right_root = (spec.repo_root / group.right).resolve()
    left_files = _iter_python_files([left_root])
    right_files = _iter_python_files([right_root])

    if right_root != left_root and right_root.is_relative_to(left_root):
        left_files = [path for path in left_files if not path.resolve().is_relative_to(right_root)]
    if left_root != right_root and left_root.is_relative_to(right_root):
        right_files = [path for path in right_files if not path.resolve().is_relative_to(left_root)]

    candidate_pairs: list[tuple[float, int, Path, Path]] = []
    normalized_sources = {path: _normalized_source(path) for path in [*left_files, *right_files]}
    for left in left_files:
        for right in right_files:
            if left.resolve() == right.resolve():
                continue
            similarity = SequenceMatcher(None, normalized_sources[left], normalized_sources[right]).ratio()
            if similarity < group.min_similarity:
                continue
            candidate_pairs.append((similarity, int(left.name == right.name), left, right))

    candidate_pairs.sort(
        key=lambda item: (
            item[0],
            item[1],
            _relative_to(item[2], spec.repo_root),
            _relative_to(item[3], spec.repo_root),
        ),
        reverse=True,
    )

    duplicate_pairs: list[DuplicateModulePair] = []
    matched_left: set[Path] = set()
    matched_right: set[Path] = set()
    for similarity, _same_name, left, right in candidate_pairs:
        if left in matched_left or right in matched_right:
            continue
        matched_left.add(left)
        matched_right.add(right)
        duplicate_pairs.append(
            DuplicateModulePair(
                group=group.name,
                left=_relative_to(left, spec.repo_root),
                right=_relative_to(right, spec.repo_root),
                similarity=round(similarity, 3),
                suggested_owner=group.canonical_owner or _find_adjacent_group_owner(spec, group.name),
            ),
        )
    return duplicate_pairs


def audit_repository(spec: AutoImproveSpec) -> AuditReport:
    """Audit the current repo state for redundancy.

    Args:
        spec: Loaded autoimprove specification.

    Returns:
        Repository redundancy audit report.
    """

    audit_roots = _resolve_paths(spec.repo_root, spec.audit_paths)
    python_files = _iter_python_files(audit_roots)

    duplicate_module_pairs: list[DuplicateModulePair] = []
    for group in spec.adjacent_module_groups:
        duplicate_module_pairs.extend(_match_adjacent_module_pairs(spec, group))

    repeated_classes: dict[str, tuple[str, list[str]]] = {}
    helper_collisions: dict[str, list[str]] = defaultdict(list)
    private_helper_modules: list[PrivateHelperModule] = []
    private_export_modules: list[PrivateExportModule] = []
    legacy_import_edges: list[LegacyImportEdge] = []

    for path in python_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_path = _relative_to(path, spec.repo_root)

        local_private_helpers = [
            node.name
            for node in _iter_top_level_functions(tree)
            if node.name.startswith("_") and node.name not in spec.ignored_helper_names
        ]
        if (
            len(local_private_helpers) >= spec.helper_density_threshold
            and not _is_under_shared_helper_root(relative_path, spec)
            and "tests/" not in relative_path
            and not relative_path.startswith("aria_nbv/tests/")
        ):
            private_helper_modules.append(
                PrivateHelperModule(
                    path=relative_path,
                    helpers=sorted(local_private_helpers),
                    suggested_owner=spec.shared_helper_roots[0] if spec.shared_helper_roots else None,
                ),
            )

        private_exports = _extract_private_exports(tree, ignored_names=spec.ignored_helper_names)
        if private_exports:
            private_export_modules.append(PrivateExportModule(path=relative_path, names=private_exports))

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                kind = _class_contract_kind(node)
                if kind is None:
                    continue
                existing = repeated_classes.get(node.name)
                if existing is None:
                    repeated_classes[node.name] = (kind, [relative_path])
                else:
                    repeated_classes[node.name] = (existing[0], [*existing[1], relative_path])
                continue

            if isinstance(node, ast.Assign):
                kind = _typed_dict_assignment_kind(node)
                if kind is None:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        existing = repeated_classes.get(target.id)
                        if existing is None:
                            repeated_classes[target.id] = (kind, [relative_path])
                        else:
                            repeated_classes[target.id] = (existing[0], [*existing[1], relative_path])

        for node in _iter_top_level_functions(tree):
            if node.name not in spec.ignored_helper_names:
                helper_collisions[node.name].append(relative_path)

        for node in ast.walk(tree):
            imported_module = ""
            imported_names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                imported_module = _resolve_import_module(relative_path, node)
                imported_names = [alias.name for alias in node.names if alias.name != "*"]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names = [alias.name]
                    imported_module = alias.name
                    for rule in spec.import_preference_rules:
                        if not (
                            imported_module == rule.legacy_prefix
                            or imported_module.startswith(f"{rule.legacy_prefix}.")
                        ):
                            continue
                        if any(_location_matches(relative_path, allowed) for allowed in rule.allow_paths):
                            continue
                        legacy_import_edges.append(
                            LegacyImportEdge(
                                importer=relative_path,
                                imported_module=imported_module,
                                names=imported_names,
                                canonical_prefix=rule.canonical_prefix,
                            ),
                        )
                    imported_module = ""
                    imported_names = []
                continue

            if not imported_module:
                continue
            for rule in spec.import_preference_rules:
                if not (imported_module == rule.legacy_prefix or imported_module.startswith(f"{rule.legacy_prefix}.")):
                    continue
                if any(_location_matches(relative_path, allowed) for allowed in rule.allow_paths):
                    continue
                legacy_import_edges.append(
                    LegacyImportEdge(
                        importer=relative_path,
                        imported_module=imported_module,
                        names=imported_names,
                        canonical_prefix=rule.canonical_prefix,
                    ),
                )

    repeated_class_groups = [
        RepeatedSymbolGroup(
            name=name,
            kind=kind,
            locations=sorted(set(locations)),
            suggested_owner=_suggest_owner(sorted(set(locations)), spec),
        )
        for name, (kind, locations) in sorted(repeated_classes.items())
        if len(set(locations)) > 1
    ]
    helper_collision_groups = [
        RepeatedSymbolGroup(
            name=name,
            kind="helper",
            locations=sorted(set(locations)),
            suggested_owner=_suggest_owner(sorted(set(locations)), spec),
        )
        for name, locations in sorted(helper_collisions.items())
        if len(set(locations)) > 1
    ]
    private_helper_modules.sort(key=lambda item: (-len(item.helpers), item.path))
    private_export_modules.sort(key=lambda item: (-len(item.names), item.path))
    legacy_import_edges.sort(key=lambda item: (item.importer, item.imported_module, item.names))

    return AuditReport(
        python_loc=_count_python_loc(python_files),
        duplicate_module_pairs=duplicate_module_pairs,
        repeated_class_groups=repeated_class_groups,
        helper_collisions=helper_collision_groups,
        private_helper_modules=private_helper_modules,
        private_export_modules=private_export_modules,
        legacy_import_edges=legacy_import_edges,
    )


def collect_diff_metrics(spec: AutoImproveSpec) -> DiffMetrics:
    """Collect git diff metrics against the configured base ref.

    Args:
        spec: Loaded autoimprove specification.

    Returns:
        Git-based diff metrics.
    """

    additions = 0
    deletions = 0
    diff_args = ["git", "diff", "--numstat", spec.diff_base, "--", *spec.editable_paths]
    try:
        numstat = subprocess.run(
            diff_args,
            cwd=spec.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        numstat = ""
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", maxsplit=2)
        if not path.endswith(".py"):
            continue
        if added.isdigit():
            additions += int(added)
        if deleted.isdigit():
            deletions += int(deleted)

    protected_touches = 0
    if spec.protected_paths:
        diff_names = ["git", "diff", "--name-only", spec.diff_base, "--", *spec.protected_paths]
        try:
            output = subprocess.run(
                diff_names,
                cwd=spec.repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError:
            output = ""
        protected_touches = len({line.strip() for line in output.splitlines() if line.strip()})

    return DiffMetrics(
        additions=additions,
        deletions=deletions,
        net_python_lines_removed=deletions - additions,
        protected_path_touches=protected_touches,
    )


def run_verify_commands(spec: AutoImproveSpec, mode: str) -> VerificationSummary:
    """Run configured verification commands for a mode.

    Args:
        spec: Loaded autoimprove specification.
        mode: Active autoimprove mode.

    Returns:
        Verification summary for the mode.
    """

    commands = spec.modes[mode].verify_commands
    if not commands:
        return VerificationSummary(pass_rate=1.0, results=[])

    results: list[VerificationCommandResult] = []
    passed = 0
    for command in commands:
        result = subprocess.run(
            ["/bin/sh", "-lc", command],
            cwd=spec.repo_root,
            check=False,
        )
        results.append(VerificationCommandResult(command=command, returncode=result.returncode))
        if result.returncode == 0:
            passed += 1
    return VerificationSummary(pass_rate=passed / len(commands), results=results)


def compute_score(
    spec: AutoImproveSpec,
    audit: AuditReport,
    diff: DiffMetrics,
    *,
    verification_pass_rate: float,
    coverage_delta: float,
) -> tuple[float, dict[str, float]]:
    """Compute the weighted autoimprove score.

    Args:
        spec: Loaded autoimprove specification.
        audit: Current redundancy audit report.
        diff: Current diff metrics.
        verification_pass_rate: Fraction of passing verification commands.
        coverage_delta: Optional measured coverage delta.

    Returns:
        Tuple of score value and feature breakdown.
    """

    features = {
        "verification_pass_rate": verification_pass_rate,
        "coverage_delta": coverage_delta,
        "net_python_lines_removed": float(diff.net_python_lines_removed),
        "duplicate_module_pairs": float(len(audit.duplicate_module_pairs)),
        "repeated_contract_groups": float(len(audit.repeated_class_groups)),
        "repeated_class_groups": float(len(audit.repeated_class_groups)),
        "helper_collisions": float(len(audit.helper_collisions)),
        "helper_dense_modules": float(len(audit.private_helper_modules)),
        "private_helper_functions": float(sum(len(item.helpers) for item in audit.private_helper_modules)),
        "private_export_names": float(sum(len(item.names) for item in audit.private_export_modules)),
        "legacy_import_edges": float(len(audit.legacy_import_edges)),
        "protected_path_touches": float(diff.protected_path_touches),
        "python_loc": float(audit.python_loc),
    }
    score = 0.0
    for name, weight in spec.cost_function.weights.items():
        score += weight * features.get(name, 0.0)
    return score, features


def _take_lines(items: list[str], limit: int = 8) -> list[str]:
    """Return at most ``limit`` items with an overflow marker.

    Args:
        items: Items to render.
        limit: Maximum number of items to include.

    Returns:
        Trimmed list of items, possibly with a final overflow marker.
    """

    if len(items) <= limit:
        return items
    overflow = len(items) - limit
    return [*items[:limit], f"... (+{overflow} more)"]


def render_prompt(spec: AutoImproveSpec, mode: str, audit: AuditReport | None = None) -> str:
    """Render a concise, mode-specific prompt from the Markdown spec.

    Args:
        spec: Loaded autoimprove specification.
        mode: Active autoimprove mode.
        audit: Optional precomputed audit report.

    Returns:
        Mode-specific prompt text.
    """

    if mode not in spec.modes:
        raise KeyError(f"Unknown mode '{mode}'. Available modes: {', '.join(sorted(spec.modes))}")
    audit = audit or audit_repository(spec)
    mode_spec = spec.modes[mode]
    duplicate_lines = [
        f"- {pair.group}: {pair.left} <-> {pair.right} ({pair.similarity:.3f}); owner={pair.suggested_owner or 'n/a'}"
        for pair in audit.duplicate_module_pairs
    ]
    repeated_type_lines = [
        f"- {group.name} [{group.kind}]: {', '.join(group.locations)}; owner={group.suggested_owner or 'n/a'}"
        for group in audit.repeated_class_groups
    ]
    helper_lines = [
        f"- {group.name}: {', '.join(group.locations)}; owner={group.suggested_owner or 'n/a'}"
        for group in audit.helper_collisions
    ]
    private_helper_lines = [
        f"- {group.path}: {', '.join(group.helpers)}; move={group.suggested_owner or 'n/a'}"
        for group in audit.private_helper_modules
    ]
    private_export_lines = [f"- {group.path}: {', '.join(group.names)}" for group in audit.private_export_modules]
    legacy_import_lines = [
        f"- {group.importer}: {group.imported_module} ({', '.join(group.names) or 'module import'}) -> {group.canonical_prefix}"
        for group in audit.legacy_import_edges
    ]
    lines = [
        f"# {spec.name}",
        "",
        f"Executor: {spec.executor}",
        f"Mode: {mode}",
        f"Goal: {mode_spec.goal}",
        f"Score to maximize: {spec.cost_function.expression}",
        "",
        "Editable paths:",
        *[f"- {path}" for path in spec.editable_paths],
        "",
        "Protected paths:",
        *[f"- {path}" for path in spec.protected_paths],
        "",
        "Current audit counts:",
        f"- duplicate module pairs: {len(audit.duplicate_module_pairs)}",
        f"- repeated contract groups: {len(audit.repeated_class_groups)}",
        f"- helper collisions: {len(audit.helper_collisions)}",
        f"- private helper modules: {len(audit.private_helper_modules)}",
        f"- private export modules: {len(audit.private_export_modules)}",
        f"- legacy import edges: {len(audit.legacy_import_edges)}",
        "",
        "Highest-overlap duplicate modules:",
        *(_take_lines(duplicate_lines) or ["- none"]),
        "",
        "Repeated contracts and models:",
        *(_take_lines(repeated_type_lines) or ["- none"]),
        "",
        "Repeated helpers:",
        *(_take_lines(helper_lines) or ["- none"]),
        "",
        "Private helper sprawl:",
        *(_take_lines(private_helper_lines) or ["- none"]),
        "",
        "Private exports:",
        *(_take_lines(private_export_lines) or ["- none"]),
        "",
        "Legacy import edges:",
        *(_take_lines(legacy_import_lines) or ["- none"]),
        "",
        "Mode focus:",
        *[f"- {item}" for item in mode_spec.focus],
        "",
        "Verification commands:",
        *[f"- {command}" for command in mode_spec.verify_commands],
        "",
        spec.prompt_body,
    ]
    return "\n".join(lines)


def _render_report(audit: AuditReport) -> str:
    """Render the audit report as Markdown.

    Args:
        audit: Current redundancy audit report.

    Returns:
        Markdown report text.
    """

    duplicate_lines = [
        f"- `{pair.group}`: `{pair.left}` <-> `{pair.right}` ({pair.similarity:.3f}), owner `{pair.suggested_owner}`"
        for pair in audit.duplicate_module_pairs
    ]
    contract_lines = [
        f"- `{group.name}` [{group.kind}]: {', '.join(f'`{item}`' for item in group.locations)}; owner `{group.suggested_owner}`"
        for group in audit.repeated_class_groups
    ]
    helper_lines = [
        f"- `{group.name}`: {', '.join(f'`{item}`' for item in group.locations)}; owner `{group.suggested_owner}`"
        for group in audit.helper_collisions
    ]
    private_helper_lines = [
        f"- `{group.path}`: {', '.join(f'`{item}`' for item in group.helpers)}; move to `{group.suggested_owner}`"
        for group in audit.private_helper_modules
    ]
    private_export_lines = [
        f"- `{group.path}`: {', '.join(f'`{item}`' for item in group.names)}" for group in audit.private_export_modules
    ]
    legacy_import_lines = [
        f"- `{group.importer}` imports `{group.imported_module}` ({', '.join(f'`{item}`' for item in group.names) or '`module import`'}) -> `{group.canonical_prefix}`"
        for group in audit.legacy_import_edges
    ]
    return "\n".join(
        [
            "# Autoimprove Audit",
            "",
            f"- Python LOC: {audit.python_loc}",
            f"- Duplicate module pairs: {len(audit.duplicate_module_pairs)}",
            f"- Repeated contract groups: {len(audit.repeated_class_groups)}",
            f"- Helper collisions: {len(audit.helper_collisions)}",
            f"- Private helper modules: {len(audit.private_helper_modules)}",
            f"- Private export modules: {len(audit.private_export_modules)}",
            f"- Legacy import edges: {len(audit.legacy_import_edges)}",
            "",
            "## Duplicate Module Pairs",
            *(duplicate_lines or ["- none"]),
            "",
            "## Repeated Contracts And Models",
            *(contract_lines or ["- none"]),
            "",
            "## Repeated Helpers",
            *(helper_lines or ["- none"]),
            "",
            "## Private Helper Sprawl",
            *(private_helper_lines or ["- none"]),
            "",
            "## Private Exports",
            *(private_export_lines or ["- none"]),
            "",
            "## Legacy Import Edges",
            *(legacy_import_lines or ["- none"]),
        ]
    )


def render_report(spec: AutoImproveSpec, audit: AuditReport, *, mode: str) -> str:
    """Render a Markdown audit report for one autoimprove mode.

    Args:
        spec: Loaded autoimprove specification.
        audit: Current redundancy audit report.
        mode: Active autoimprove mode.

    Returns:
        Markdown report text.
    """

    del spec, mode
    return _render_report(audit)


def _report_to_json(
    audit: AuditReport, diff: DiffMetrics | None = None, score_payload: dict[str, Any] | None = None
) -> str:
    """Serialize audit and optional score state to JSON.

    Args:
        audit: Current redundancy audit report.
        diff: Optional diff metrics.
        score_payload: Optional score breakdown.

    Returns:
        JSON string.
    """

    payload: dict[str, Any] = {"audit": asdict(audit)}
    if diff is not None:
        payload["diff"] = asdict(diff)
    if score_payload is not None:
        payload["score"] = score_payload
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the autoimprove utility.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process return code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=None, help="Path to autoimprove.md. Defaults to the nearest parent file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show", help="Show the resolved autoimprove spec.")

    audit_parser = subparsers.add_parser("audit", help="Audit the repo for duplication.")
    audit_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    report_parser = subparsers.add_parser("report", help="Render the full audit report.")
    report_parser.add_argument("--mode", default=None, help="Optional mode to annotate in the report header.")
    report_parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    report_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    prompt_parser = subparsers.add_parser("prompt", help="Render the mode-specific prompt.")
    prompt_parser.add_argument("--mode", default=None, help="Autoimprove mode to render.")

    score_parser = subparsers.add_parser("score", help="Score the current repo state.")
    score_parser.add_argument("--mode", default=None, help="Autoimprove mode used for verification commands.")
    score_parser.add_argument(
        "--run-verify", action="store_true", help="Run mode verification commands before scoring."
    )
    score_parser.add_argument("--coverage-delta", type=float, default=0.0, help="Optional measured coverage delta.")
    score_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    args = parser.parse_args(argv)
    spec = load_autoimprove_spec(args.spec)

    if args.command == "show":
        print(json.dumps(asdict(spec), indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "audit":
        audit = audit_repository(spec)
        if args.json:
            print(_report_to_json(audit))
        else:
            print(f"duplicate module pairs: {len(audit.duplicate_module_pairs)}")
            print(f"repeated contract groups: {len(audit.repeated_class_groups)}")
            print(f"helper collisions: {len(audit.helper_collisions)}")
            print(f"private helper modules: {len(audit.private_helper_modules)}")
            print(f"private export modules: {len(audit.private_export_modules)}")
            print(f"legacy import edges: {len(audit.legacy_import_edges)}")
        return 0

    if args.command == "report":
        audit = audit_repository(spec)
        if args.json:
            print(_report_to_json(audit))
        else:
            mode = args.mode or spec.default_mode
            report = render_report(spec, audit, mode=mode)
            if args.output:
                output_path = (spec.repo_root / args.output).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(report)
                print(output_path.relative_to(spec.repo_root).as_posix())
            else:
                print(report)
        return 0

    if args.command == "prompt":
        mode = args.mode or spec.default_mode
        print(render_prompt(spec, mode))
        return 0

    if args.command == "score":
        mode = args.mode or spec.default_mode
        audit = audit_repository(spec)
        diff = collect_diff_metrics(spec)
        verification_pass_rate = 1.0
        verification_results: list[VerificationCommandResult] = []
        if args.run_verify:
            verification = run_verify_commands(spec, mode)
            verification_pass_rate = verification.pass_rate
            verification_results = verification.results
        score, features = compute_score(
            spec,
            audit,
            diff,
            verification_pass_rate=verification_pass_rate,
            coverage_delta=args.coverage_delta,
        )
        payload = {
            "value": score,
            "features": features,
            "mode": mode,
            "verification_results": [asdict(item) for item in verification_results],
        }
        if args.json:
            print(_report_to_json(audit, diff=diff, score_payload=payload))
        else:
            print(f"score: {score:.3f}")
            for name, value in features.items():
                print(f"{name}: {value}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

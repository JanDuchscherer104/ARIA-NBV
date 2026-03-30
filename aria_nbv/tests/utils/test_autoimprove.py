"""Tests for the repo-local autoimprove helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from aria_nbv.utils.autoimprove import (
    AuditReport,
    DiffMetrics,
    DuplicateModulePair,
    LegacyImportEdge,
    PrivateExportModule,
    PrivateHelperModule,
    RepeatedSymbolGroup,
    audit_repository,
    compute_score,
    load_autoimprove_spec,
    render_prompt,
    render_report,
)


def _write_spec(path: Path, body: str) -> None:
    path.write_text(body)


def test_load_autoimprove_spec_parses_markdown_front_matter(tmp_path: Path) -> None:
    spec_path = tmp_path / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
executor: codex
report_dir: .agents/workspace/autoimprove/reports
diff_base: main
audit_paths:
  - pkg
editable_paths:
  - pkg
protected_paths:
  - generated
adjacent_module_groups:
  - name: pair
    left: pkg/a
    right: pkg/b
    min_similarity: 0.8
    canonical_owner: pkg/b
canonical_owner_rules:
  - name: prefer_b
    matches:
      - pkg/a
      - pkg/b
    prefer: pkg/b
shared_helper_roots:
  - pkg/utils
ignored_helper_names:
  - main
helper_density_threshold: 3
cost_function:
  objective: maximize
  expression: score
  weights:
    duplicate_module_pairs: -1
modes:
  simplify:
    goal: reduce duplication
    focus:
      - wrappers
    verify_commands:
      - pytest
default_mode: simplify
---
Prompt body.
""",
    )

    spec = load_autoimprove_spec(spec_path)

    assert spec.name == "demo"
    assert spec.executor == "codex"
    assert spec.report_dir == ".agents/workspace/autoimprove/reports"
    assert spec.diff_base == "main"
    assert spec.audit_paths == ["pkg"]
    assert spec.default_mode == "simplify"
    assert spec.helper_density_threshold == 3
    assert spec.adjacent_module_groups[0].left == "pkg/a"
    assert spec.adjacent_module_groups[0].canonical_owner == "pkg/b"
    assert spec.canonical_owner_rules[0].prefer == "pkg/b"
    assert spec.modes["simplify"].verify_commands == ["pytest"]
    assert spec.prompt_body == "Prompt body."


def test_render_prompt_includes_mode_goal_and_cost_function(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "pkg").mkdir()
    (repo_root / "pkg" / "one.py").write_text("class One:\n    pass\n")
    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
executor: codex
report_dir: .agents/workspace/autoimprove/reports
audit_paths:
  - pkg
editable_paths:
  - pkg
protected_paths: []
adjacent_module_groups: []
canonical_owner_rules: []
shared_helper_roots: []
ignored_helper_names: []
helper_density_threshold: 2
cost_function:
  objective: maximize
  expression: score = good - bad
  weights:
    duplicate_module_pairs: -1
modes:
  simplify:
    goal: reduce duplication
    focus:
      - single owner
    verify_commands:
      - pytest
default_mode: simplify
---
Body text.
""",
    )

    prompt = render_prompt(load_autoimprove_spec(spec_path), "simplify")

    assert "Mode: simplify" in prompt
    assert "Executor: codex" in prompt
    assert "Goal: reduce duplication" in prompt
    assert "Score to maximize: score = good - bad" in prompt
    assert "Body text." in prompt


def test_audit_detects_near_duplicate_adjacent_modules(tmp_path: Path) -> None:
    repo_root = tmp_path
    left_root = repo_root / "pkg" / "left"
    right_root = repo_root / "pkg" / "right"
    left_root.mkdir(parents=True)
    right_root.mkdir(parents=True)
    source = """class SharedConfig:\n    value: int = 1\n\ndef helper() -> int:\n    return 1\n"""
    (left_root / "shared.py").write_text(source)
    (right_root / "shared.py").write_text(source)

    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
executor: codex
report_dir: .agents/workspace/autoimprove/reports
audit_paths:
  - pkg
editable_paths:
  - pkg
protected_paths: []
adjacent_module_groups:
  - name: pair
    left: pkg/left
    right: pkg/right
    min_similarity: 0.95
    canonical_owner: pkg/right
canonical_owner_rules:
  - name: prefer_right
    matches:
      - pkg/left
      - pkg/right
    prefer: pkg/right
shared_helper_roots:
  - pkg/shared_utils
ignored_helper_names: []
helper_density_threshold: 2
cost_function:
  objective: maximize
  expression: score
  weights:
    duplicate_module_pairs: -1
    repeated_class_groups: -1
    helper_collisions: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )

    spec = load_autoimprove_spec(spec_path)
    audit = audit_repository(spec)

    assert len(audit.duplicate_module_pairs) == 1
    assert audit.duplicate_module_pairs[0].group == "pair"
    assert audit.duplicate_module_pairs[0].suggested_owner == "pkg/right"
    assert any(group.name == "SharedConfig" for group in audit.repeated_class_groups)
    assert any(group.name == "helper" for group in audit.helper_collisions)
    assert any(
        group.name == "SharedConfig" and group.suggested_owner == "pkg/right/shared.py"
        for group in audit.repeated_class_groups
    )


def test_compute_score_uses_configured_weights() -> None:
    spec = load_autoimprove_spec(Path(__file__).resolve().parents[3] / "autoimprove.md")
    audit = AuditReport(
        python_loc=100,
        duplicate_module_pairs=[DuplicateModulePair("g", "a", "b", 0.9)],
        repeated_class_groups=[RepeatedSymbolGroup("A", "contract", ["x", "y"])],
        helper_collisions=[],
        private_helper_modules=[],
        private_export_modules=[],
        legacy_import_edges=[],
    )
    diff = DiffMetrics(
        additions=10,
        deletions=50,
        net_python_lines_removed=40,
        protected_path_touches=0,
    )

    score, features = compute_score(
        spec,
        audit,
        diff,
        verification_pass_rate=1.0,
        coverage_delta=0.0,
    )

    weights = spec.cost_function.weights
    expected = (
        weights["verification_pass_rate"] * 1.0
        + weights["coverage_delta"] * 0.0
        + weights["net_python_lines_removed"] * 40.0
        + weights["duplicate_module_pairs"] * 1.0
        + weights["repeated_class_groups"] * 1.0
        + weights["helper_collisions"] * 0.0
        + weights["private_helper_functions"] * 0.0
        + weights["private_export_names"] * 0.0
        + weights["legacy_import_edges"] * 0.0
        + weights["protected_path_touches"] * 0.0
    )
    assert score == expected
    assert features["net_python_lines_removed"] == 40.0


def test_audit_matches_renamed_adjacent_duplicate_modules(tmp_path: Path) -> None:
    repo_root = tmp_path
    left_root = repo_root / "pkg" / "legacy"
    right_root = repo_root / "pkg" / "new"
    left_root.mkdir(parents=True)
    right_root.mkdir(parents=True)
    source = """class CacheConfig:\n    value: int = 1\n\ndef _read_index() -> int:\n    return 1\n"""
    (left_root / "offline_cache.py").write_text(source)
    (right_root / "oracle_cache.py").write_text(source)

    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
editable_paths:
  - pkg
audit_paths:
  - pkg
protected_paths: []
adjacent_module_groups:
  - name: pair
    left: pkg/legacy
    right: pkg/new
    min_similarity: 0.95
    canonical_owner: pkg/new
canonical_owner_rules:
  - name: prefer_new
    matches:
      - pkg/legacy
      - pkg/new
    prefer: pkg/new
shared_helper_roots: []
ignored_helper_names: []
helper_density_threshold: 2
cost_function:
  objective: maximize
  expression: score
  weights:
    duplicate_module_pairs: -1
    repeated_contract_groups: -1
    helper_collisions: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )

    audit = audit_repository(load_autoimprove_spec(spec_path))

    assert len(audit.duplicate_module_pairs) == 1
    assert audit.duplicate_module_pairs[0].left.endswith("offline_cache.py")
    assert audit.duplicate_module_pairs[0].right.endswith("oracle_cache.py")
    assert audit.duplicate_module_pairs[0].suggested_owner == "pkg/new"


def test_render_report_lists_helper_dense_modules(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "pkg").mkdir()
    (repo_root / "pkg" / "feature.py").write_text(
        "\n".join(
            [
                "def _one() -> int:",
                "    return 1",
                "",
                "def _two() -> int:",
                "    return 2",
                "",
                "def _three() -> int:",
                "    return 3",
            ],
        ),
    )
    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
editable_paths:
  - pkg
audit_paths:
  - pkg
protected_paths: []
adjacent_module_groups: []
shared_helper_roots: []
ignored_helper_names: []
helper_density_threshold: 2
cost_function:
  objective: maximize
  expression: score
  weights:
    private_helper_functions: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )

    spec = load_autoimprove_spec(spec_path)
    audit = audit_repository(spec)
    report = render_report(spec, audit, mode="simplify")

    assert audit.private_helper_modules[0].path == "pkg/feature.py"
    assert "Private Helper Sprawl" in report
    assert "`pkg/feature.py`" in report


def test_audit_flags_private_helper_modules_outside_shared_roots(tmp_path: Path) -> None:
    repo_root = tmp_path
    feature_dir = repo_root / "pkg" / "feature"
    shared_dir = repo_root / "pkg" / "utils"
    feature_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)
    (feature_dir / "worker.py").write_text(
        """def _one() -> int:\n    return 1\n\n\ndef _two() -> int:\n    return 2\n"""
    )
    (shared_dir / "shared.py").write_text("""def _ok() -> int:\n    return 1\n""")

    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
editable_paths:
  - pkg
audit_paths:
  - pkg
protected_paths: []
adjacent_module_groups: []
canonical_owner_rules: []
shared_helper_roots:
  - pkg/utils
ignored_helper_names: []
helper_density_threshold: 2
cost_function:
  objective: maximize
  expression: score
  weights:
    private_helper_functions: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )

    audit = audit_repository(load_autoimprove_spec(spec_path))

    assert len(audit.private_helper_modules) == 1
    assert audit.private_helper_modules[0] == PrivateHelperModule(
        path="pkg/feature/worker.py",
        helpers=["_one", "_two"],
        suggested_owner="pkg/utils",
    )


def test_audit_flags_private_exports_and_legacy_import_edges(tmp_path: Path) -> None:
    repo_root = tmp_path
    pkg_dir = repo_root / "pkg"
    legacy_dir = pkg_dir / "legacy"
    consumer_dir = pkg_dir / "app"
    legacy_dir.mkdir(parents=True)
    consumer_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (legacy_dir / "__init__.py").write_text("")
    (consumer_dir / "__init__.py").write_text("")
    (legacy_dir / "helpers.py").write_text(
        "__all__ = ['_hidden', 'public']\n\n_hidden = 1\npublic = 2\n",
    )
    (consumer_dir / "panel.py").write_text(
        "from ..legacy.helpers import public\n\nvalue = public\n",
    )

    spec_path = repo_root / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
editable_paths:
  - pkg
audit_paths:
  - pkg
protected_paths: []
adjacent_module_groups: []
canonical_owner_rules: []
import_preference_rules:
  - name: prefer_new_helpers
    legacy_prefix: pkg.legacy
    canonical_prefix: pkg.shared
    allow_paths:
      - pkg/legacy
shared_helper_roots: []
ignored_helper_names: []
helper_density_threshold: 3
cost_function:
  objective: maximize
  expression: score
  weights:
    private_export_names: -1
    legacy_import_edges: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )

    audit = audit_repository(load_autoimprove_spec(spec_path))

    assert audit.private_export_modules == [
        PrivateExportModule(path="pkg/legacy/helpers.py", names=["_hidden"]),
    ]
    assert audit.legacy_import_edges == [
        LegacyImportEdge(
            importer="pkg/app/panel.py",
            imported_module="pkg.legacy.helpers",
            names=["public"],
            canonical_prefix="pkg.shared",
        ),
    ]


def test_module_cli_prints_report(tmp_path: Path) -> None:
    spec_path = tmp_path / "autoimprove.md"
    _write_spec(
        spec_path,
        """---
name: demo
version: 1
editable_paths:
  - pkg
audit_paths:
  - pkg
protected_paths: []
adjacent_module_groups: []
canonical_owner_rules: []
shared_helper_roots: []
ignored_helper_names: []
cost_function:
  objective: maximize
  expression: score
  weights:
    duplicate_module_pairs: -1
modes:
  simplify:
    goal: reduce duplication
default_mode: simplify
---
Body.
""",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "one.py").write_text("class One:\n    pass\n")

    result = subprocess.run(
        [sys.executable, "-m", "aria_nbv.utils.autoimprove", "--spec", str(spec_path), "report"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Autoimprove Audit" in result.stdout

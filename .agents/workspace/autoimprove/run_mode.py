"""Run one bounded Aria-NBV autoimprove mode and emit a report.

This script is the operator-facing wrapper around
``aria_nbv.utils.autoimprove``. It mirrors the spirit of
``karpathy/autoresearch`` while keeping the loop bounded and code-reviewable:
one audit, one selected mode, one score, one report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aria_nbv.utils.autoimprove import (
    audit_repository,
    collect_diff_metrics,
    compute_score,
    load_autoimprove_spec,
    render_prompt,
    run_verify_commands,
)


def _repo_root() -> Path:
    """Return the repository root inferred from this script location."""
    return Path(__file__).resolve().parents[3]


def _default_spec_path() -> Path:
    """Return the default autoimprove spec path."""
    return _repo_root() / "autoimprove.md"


def _render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a human-readable Markdown report for one bounded pass.

    Args:
        payload: Structured report payload assembled by ``main``.

    Returns:
        Markdown report content.
    """
    audit = payload["audit"]
    diff = payload["diff"]
    score = payload["score"]
    verification = payload["verification"]

    lines = [
        f"# Autoimprove Report: {payload['mode']}",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Spec: `{payload['spec_path']}`",
        f"- Executor: `{payload['executor']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Diff base: `{payload['diff_base']}`",
        "",
        "## Objective",
        "",
        payload["cost_expression"],
        "",
        "## Score",
        "",
        f"- Value: `{score['value']:.3f}`",
        "",
        "### Features",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in score["features"].items())
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- Pass rate: `{verification['pass_rate']:.3f}`",
        ],
    )
    lines.extend(
        f"- `{item['command']}` -> `{item['returncode']}`"
        for item in verification["results"]
    )
    lines.extend(
        [
            "",
            "## Audit Summary",
            "",
            f"- Python LOC: `{audit['python_loc']}`",
            f"- Duplicate module pairs: `{len(audit['duplicate_module_pairs'])}`",
            f"- Repeated class groups: `{len(audit['repeated_class_groups'])}`",
            f"- Helper collisions: `{len(audit['helper_collisions'])}`",
            "",
            "## Duplicate Module Pairs",
            "",
        ],
    )
    lines.extend(
        (
            f"- `{pair['group']}`: `{pair['left']}` <-> "
            f"`{pair['right']}` (`{pair['similarity']}`)"
        )
        for pair in audit["duplicate_module_pairs"]
    )
    lines.extend(["", "## Repeated Class Groups", ""])
    lines.extend(
        f"- `{group['name']}`: "
        + ", ".join(f"`{location}`" for location in group["locations"])
        for group in audit["repeated_class_groups"]
    )
    lines.extend(["", "## Helper Collisions", ""])
    lines.extend(
        f"- `{group['name']}`: "
        + ", ".join(f"`{location}`" for location in group["locations"])
        for group in audit["helper_collisions"]
    )
    lines.extend(
        [
            "",
            "## Diff Metrics",
            "",
            f"- Additions: `{diff['additions']}`",
            f"- Deletions: `{diff['deletions']}`",
            f"- Net Python lines removed: `{diff['net_python_lines_removed']}`",
            f"- Protected path touches: `{diff['protected_path_touches']}`",
            "",
            "## Prompt",
            "",
            "```markdown",
            payload["prompt"].rstrip(),
            "```",
        ],
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run one autoimprove mode and write a report.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=_default_spec_path(),
        help="Path to the GT autoimprove spec.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode to run. Defaults to the spec's default mode.",
    )
    parser.add_argument(
        "--run-verify",
        action="store_true",
        help="Execute the selected mode's verification commands before scoring.",
    )
    args = parser.parse_args(argv)

    spec = load_autoimprove_spec(args.spec)
    mode = args.mode or spec.default_mode
    audit = audit_repository(spec)
    diff = collect_diff_metrics(spec)
    verification = run_verify_commands(spec, mode) if args.run_verify else None
    verification_pass_rate = 1.0 if verification is None else verification.pass_rate
    score_value, features = compute_score(
        spec,
        audit,
        diff,
        verification_pass_rate=verification_pass_rate,
        coverage_delta=0.0,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "mode": mode,
        "executor": spec.executor,
        "diff_base": spec.diff_base,
        "cost_expression": spec.cost_function.expression,
        "spec_path": str(spec.path.relative_to(spec.repo_root)),
        "audit": asdict(audit),
        "diff": asdict(diff),
        "score": {
            "value": score_value,
            "features": features,
        },
        "verification": {
            "pass_rate": verification_pass_rate,
            "results": []
            if verification is None
            else [asdict(item) for item in verification.results],
        },
        "prompt": render_prompt(spec, mode, audit=audit),
    }

    reports_dir = spec.repo_root / spec.report_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{mode}"
    markdown_path = reports_dir / f"{stem}.md"
    json_path = reports_dir / f"{stem}.json"
    markdown_path.write_text(_render_markdown_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(markdown_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

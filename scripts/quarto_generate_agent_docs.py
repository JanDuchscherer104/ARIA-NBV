#!/usr/bin/env python3
"""Generate Quarto pages for maintained agent guidance and scaffold docs."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
OUTPUT_ROOT = DOCS_ROOT / "contents" / "resources" / "agent_scaffold"
GITHUB_BLOB_BASE = "https://github.com/JanDuchscherer104/NBV/blob/main"
GENERATED_NOTE = (
    "This page is generated from the canonical repo markdown source. "
    "Refresh it with `./scripts/quarto_generate_agent_docs.py`."
)


@dataclass(frozen=True)
class DocSpec:
    source: str
    output: str
    title: str
    section: str
    nav_label: str
    summary: str


DOC_SPECS: tuple[DocSpec, ...] = (
    DocSpec(
        source="AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/repo_guidance.qmd",
        title="Repo Guidance",
        section="Instructions",
        nav_label="Repo Guidance",
        summary="Root Codex guidance, bootstrap order, commands, and repo-wide constraints.",
    ),
    DocSpec(
        source="docs/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/docs_guidance.qmd",
        title="Docs Guidance",
        section="Instructions",
        nav_label="Docs Guidance",
        summary="Quarto and Typst editing rules for the docs workspace.",
    ),
    DocSpec(
        source="aria_nbv/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/package_guidance.qmd",
        title="Package Guidance",
        section="Instructions",
        nav_label="Package Guidance",
        summary="Package-level implementation, verification, and config-factory rules.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/app/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/app_guidance.qmd",
        title="App Guidance",
        section="Instructions",
        nav_label="App Guidance",
        summary="Streamlit app, panel, state, and UI guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/configs/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/configs_guidance.qmd",
        title="Config Guidance",
        section="Instructions",
        nav_label="Config Guidance",
        summary="Persistent config, path, W&B, and Optuna guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/data_handling/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/data_handling_boundary.qmd",
        title="Data Handling Boundary",
        section="Instructions",
        nav_label="Data Handling Boundary",
        summary="Contracts and verification rules for the typed data-handling surface.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/lightning/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/lightning_guidance.qmd",
        title="Lightning Guidance",
        section="Instructions",
        nav_label="Lightning Guidance",
        summary="Training experiment, datamodule, module, trainer, and CLI guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/pipelines/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/pipelines_guidance.qmd",
        title="Pipeline Guidance",
        section="Instructions",
        nav_label="Pipeline Guidance",
        summary="Pipeline entrypoint and oracle-labeler orchestration guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/pose_generation/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/pose_generation_guidance.qmd",
        title="Pose Generation Guidance",
        section="Instructions",
        nav_label="Pose Generation Guidance",
        summary="Candidate generation, feasibility, orientation, and counterfactual pose guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/rendering/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/rendering_guidance.qmd",
        title="Rendering Guidance",
        section="Instructions",
        nav_label="Rendering Guidance",
        summary="Depth rendering, unprojection, point-cloud, and diagnostic guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/rl/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/rl_guidance.qmd",
        title="RL Guidance",
        section="Instructions",
        nav_label="RL Guidance",
        summary="Discrete-shell RL and counterfactual environment guidance.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/rri_metrics/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/rri_metrics_boundary.qmd",
        title="RRI Metrics Boundary",
        section="Instructions",
        nav_label="RRI Metrics Boundary",
        summary="Boundary rules for oracle RRI semantics, binning, and related metrics.",
    ),
    DocSpec(
        source="aria_nbv/aria_nbv/vin/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/vin_boundary.qmd",
        title="VIN Boundary",
        section="Instructions",
        nav_label="VIN Boundary",
        summary="Guidance for scorer inputs, batch contracts, and VIN-facing frame semantics.",
    ),
    DocSpec(
        source="docs/typst/paper/AGENTS.md",
        output="contents/resources/agent_scaffold/instructions/typst_paper_guidance.qmd",
        title="Typst Paper Guidance",
        section="Instructions",
        nav_label="Typst Paper Guidance",
        summary="Paper-specific Typst and manuscript guidance.",
    ),
    DocSpec(
        source=".agents/memory/README.md",
        output="contents/resources/agent_scaffold/state/memory_readme.qmd",
        title="Agent Memory README",
        section="Canonical State",
        nav_label="Memory README",
        summary="Overview of the canonical memory layout and how history is organized.",
    ),
    DocSpec(
        source=".agents/memory/state/PROJECT_STATE.md",
        output="contents/resources/agent_scaffold/state/project_state.qmd",
        title="Project State",
        section="Canonical State",
        nav_label="Project State",
        summary="Current project truth for goals, architecture, and active working assumptions.",
    ),
    DocSpec(
        source=".agents/memory/state/DECISIONS.md",
        output="contents/resources/agent_scaffold/state/decisions.qmd",
        title="Decisions",
        section="Canonical State",
        nav_label="Decisions",
        summary="Durable repo and technical decisions that shape the active scaffold.",
    ),
    DocSpec(
        source=".agents/memory/state/OPEN_QUESTIONS.md",
        output="contents/resources/agent_scaffold/state/open_questions.qmd",
        title="Open Questions",
        section="Canonical State",
        nav_label="Open Questions",
        summary="Tracked unknowns, pending experiments, and unresolved design questions.",
    ),
    DocSpec(
        source=".agents/memory/state/GOTCHAS.md",
        output="contents/resources/agent_scaffold/state/gotchas.qmd",
        title="Gotchas",
        section="Canonical State",
        nav_label="Gotchas",
        summary="Maintained pitfalls, validation traps, and environment caveats.",
    ),
    DocSpec(
        source=".agents/references/operator_quick_reference.md",
        output="contents/resources/agent_scaffold/references/operator_quick_reference.qmd",
        title="Operator Quick Reference",
        section="References",
        nav_label="Operator Quick Reference",
        summary="Compact operator aid for environment recovery, repo hygiene, and key commands.",
    ),
    DocSpec(
        source=".agents/references/python_conventions.md",
        output="contents/resources/agent_scaffold/references/python_conventions.qmd",
        title="Python Conventions",
        section="References",
        nav_label="Python Conventions",
        summary="Long-form Python style, typing, docstring, and config examples.",
    ),
    DocSpec(
        source=".agents/references/context7_library_ids.md",
        output="contents/resources/agent_scaffold/references/context7_library_ids.qmd",
        title="Context7 Library IDs",
        section="References",
        nav_label="Context7 Library IDs",
        summary="Lookup table for approved Context7 library identifiers used in repo workflows.",
    ),
    DocSpec(
        source=".agents/references/agent_memory_templates.md",
        output="contents/resources/agent_scaffold/references/agent_memory_templates.qmd",
        title="Agent Memory Templates",
        section="References",
        nav_label="Agent Memory Templates",
        summary="Canonical frontmatter and body templates for native debrief records.",
    ),
    DocSpec(
        source=".agents/references/gitnexus_optional.md",
        output="contents/resources/agent_scaffold/references/gitnexus_optional.qmd",
        title="Optional GitNexus Workflow",
        section="References",
        nav_label="Optional GitNexus Workflow",
        summary="Optional GitNexus workflow and fallback impact-analysis guidance.",
    ),
    DocSpec(
        source=".agents/references/codex_hooks.md",
        output="contents/resources/agent_scaffold/references/codex_hooks.qmd",
        title="Codex Hook Templates",
        section="References",
        nav_label="Codex Hook Templates",
        summary="Inactive hook-template guidance and activation notes.",
    ),
    DocSpec(
        source=".agents/skills/aria-nbv-context/SKILL.md",
        output="contents/resources/agent_scaffold/skills/aria_nbv_context_skill.qmd",
        title="aria-nbv-context Skill",
        section="Skills & Routing",
        nav_label="aria-nbv-context Skill",
        summary="Lightweight repo-routing entrypoint for broad tasks.",
    ),
    DocSpec(
        source=".agents/skills/aria-nbv-docs-context/SKILL.md",
        output="contents/resources/agent_scaffold/skills/aria_nbv_docs_context_skill.qmd",
        title="aria-nbv-docs-context Skill",
        section="Skills & Routing",
        nav_label="aria-nbv-docs-context Skill",
        summary="Docs, Typst, Quarto, bibliography, and literature context workflow.",
    ),
    DocSpec(
        source=".agents/skills/aria-nbv-code-context/SKILL.md",
        output="contents/resources/agent_scaffold/skills/aria_nbv_code_context_skill.qmd",
        title="aria-nbv-code-context Skill",
        section="Skills & Routing",
        nav_label="aria-nbv-code-context Skill",
        summary="Python package contract and symbol-localization workflow.",
    ),
    DocSpec(
        source=".agents/skills/aria-nbv-scaffold-maintenance/SKILL.md",
        output="contents/resources/agent_scaffold/skills/aria_nbv_scaffold_maintenance_skill.qmd",
        title="aria-nbv-scaffold-maintenance Skill",
        section="Skills & Routing",
        nav_label="aria-nbv-scaffold-maintenance Skill",
        summary="Agent scaffold maintenance, validation, hooks, and DB workflow.",
    ),
    DocSpec(
        source=".agents/skills/aria-nbv-context/references/context_map.md",
        output="contents/resources/agent_scaffold/skills/context_map.qmd",
        title="Context Map",
        section="Skills & Routing",
        nav_label="Context Map",
        summary="Concept-to-source routing map for the maintained scaffold surfaces.",
    ),
    DocSpec(
        source=".agents/AGENTS_INTERNAL_DB.md",
        output="contents/resources/agent_scaffold/db/agents_internal_db.qmd",
        title="AGENTS Internal Database",
        section="Agent DB",
        nav_label="AGENTS Internal DB",
        summary="Stable agent-scaffold facts and DB scope.",
    ),
    DocSpec(
        source=".agents/issues.toml",
        output="contents/resources/agent_scaffold/db/issues.qmd",
        title="Agent Issues",
        section="Agent DB",
        nav_label="Agent Issues",
        summary="Active agent/tooling issues tracked by the local DB.",
    ),
    DocSpec(
        source=".agents/todos.toml",
        output="contents/resources/agent_scaffold/db/todos.qmd",
        title="Agent Todos",
        section="Agent DB",
        nav_label="Agent Todos",
        summary="Active agent/tooling todos tracked by the local DB.",
    ),
    DocSpec(
        source=".agents/resolved.toml",
        output="contents/resources/agent_scaffold/db/resolved.qmd",
        title="Resolved Agent DB",
        section="Agent DB",
        nav_label="Resolved Agent DB",
        summary="Resolved or retired agent/tooling DB records.",
    ),
)

SECTION_ORDER = (
    "Instructions",
    "Canonical State",
    "References",
    "Skills & Routing",
    "Agent DB",
)

SITE_ASSET_SUFFIXES = {".qmd", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".gif"}
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    match = re.match(r"^---\n(.*?)\n---\n?", text, flags=re.DOTALL)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, text[match.end() :]


def split_heading(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        return title, "\n".join(lines[1:]).lstrip("\n")
    return None, text.lstrip("\n")


def relative_path(source: Path, target: Path) -> str:
    return Path(os.path.relpath(target, source.parent)).as_posix()


def to_repo_relative(path: Path) -> str | None:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def translate_target(
    target: str,
    *,
    source_file: Path,
    output_file: Path,
    source_map: dict[str, DocSpec],
) -> str:
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return target

    target_path, fragment = target, ""
    if "#" in target:
        target_path, fragment = target.split("#", 1)
        fragment = f"#{fragment}"

    resolved = (source_file.parent / target_path).resolve()
    repo_relative = to_repo_relative(resolved)
    if repo_relative is None or not resolved.exists():
        return target

    if repo_relative in source_map:
        mapped = DOCS_ROOT / source_map[repo_relative].output
        return f"{relative_path(output_file, mapped)}{fragment}"

    if repo_relative.startswith("docs/"):
        docs_relative = Path(repo_relative).relative_to("docs")
        if docs_relative.suffix.lower() in SITE_ASSET_SUFFIXES:
            mapped = DOCS_ROOT / docs_relative
            return f"{relative_path(output_file, mapped)}{fragment}"

    return f"{GITHUB_BLOB_BASE}/{repo_relative}{fragment}"


def rewrite_links(
    markdown: str,
    *,
    source_file: Path,
    output_file: Path,
    source_map: dict[str, DocSpec],
) -> str:
    rewritten_lines: list[str] = []
    in_fence = False

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            rewritten_lines.append(line)
            continue
        if in_fence:
            rewritten_lines.append(line)
            continue

        def replace(match: re.Match[str]) -> str:
            label = match.group(1)
            target = match.group(2).strip()
            translated = translate_target(
                target,
                source_file=source_file,
                output_file=output_file,
                source_map=source_map,
            )
            return f"[{label}]({translated})"

        rewritten_lines.append(LINK_RE.sub(replace, line))

    return "\n".join(rewritten_lines).rstrip() + "\n"


def render_page(spec: DocSpec, source_map: dict[str, DocSpec]) -> None:
    source_file = REPO_ROOT / spec.source
    output_file = DOCS_ROOT / spec.output
    raw_text = source_file.read_text(encoding="utf-8")
    if source_file.suffix == ".toml":
        metadata: dict[str, str] = {}
        source_heading = spec.title
        body = f"```toml\n{raw_text.rstrip()}\n```"
    else:
        metadata, text = split_frontmatter(raw_text)
        source_heading, body = split_heading(text)
        body = rewrite_links(
            body,
            source_file=source_file,
            output_file=output_file,
            source_map=source_map,
        )

    callout_lines = [
        '::: {.callout-note collapse="true"}',
        "## Canonical Source",
        f"- Source file: [`{spec.source}`]({GITHUB_BLOB_BASE}/{spec.source})",
        f"- Published page: `{spec.output}`",
        f"- Refresh: `./scripts/quarto_generate_agent_docs.py`",
    ]
    if "scope" in metadata:
        callout_lines.append(f"- Scope: `{metadata['scope']}`")
    if "applies_to" in metadata:
        callout_lines.append(f"- Applies to: `{metadata['applies_to']}`")
    summary = metadata.get("summary", spec.summary)
    if summary:
        callout_lines.append(f"- Summary: {summary}")
    callout_lines.append(":::")

    page_title = source_heading or spec.title
    output_text = [
        "---",
        f"title: {yaml_quote(page_title)}",
        "format: html",
        "---",
        "",
        f"<!-- Generated by scripts/quarto_generate_agent_docs.py from {spec.source}. -->",
        "",
        GENERATED_NOTE,
        "",
        *callout_lines,
        "",
        body.strip(),
        "",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(output_text), encoding="utf-8")


def render_index() -> None:
    grouped: dict[str, list[DocSpec]] = {section: [] for section in SECTION_ORDER}
    for spec in DOC_SPECS:
        grouped[spec.section].append(spec)

    lines = [
        "---",
        'title: "Agent Scaffolding"',
        "format: html",
        "---",
        "",
        "<!-- Generated by scripts/quarto_generate_agent_docs.py. -->",
        "",
        "This section publishes the maintained Codex guidance and scaffold markdown",
        "surfaces from the repository under Quarto `Resources`.",
        "",
        "It includes the active `AGENTS.md` files, canonical memory state, agent",
        "references, repo-local skills, and the active agent DB. It intentionally",
        "excludes episodic history, `.agents/archive/`, and temporary workpads.",
        "",
        "Refresh these pages with `./scripts/quarto_generate_agent_docs.py`. The",
        "GitHub Pages workflow runs that generator before rendering the site.",
        "",
    ]

    for section in SECTION_ORDER:
        lines.extend(
            [
                f"## {section}",
                "",
            ]
        )
        for spec in grouped[section]:
            output_path = Path(spec.output).relative_to(
                "contents/resources/agent_scaffold"
            )
            lines.append(
                f"- [{spec.nav_label}]({output_path.as_posix()}): {spec.summary}"
            )
        lines.append("")

    index_file = OUTPUT_ROOT / "index.qmd"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_ROOT.iterdir():
        if child.name == ".gitignore":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    source_map = {spec.source: spec for spec in DOC_SPECS}
    for spec in DOC_SPECS:
        render_page(spec, source_map)
    render_index()


if __name__ == "__main__":
    main()

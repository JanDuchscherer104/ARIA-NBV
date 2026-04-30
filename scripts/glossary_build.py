#!/usr/bin/env python3
"""Build ARIA-NBV glossary artifacts from the canonical YAML source."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERMS = ROOT / "docs/glossary/terms.yml"
DEFAULT_QMD = ROOT / "docs/contents/glossary.qmd"
DEFAULT_TYPST = ROOT / "docs/typst/shared/glossary.generated.typ"
DEFAULT_JSONL = ROOT / "docs/_generated/context/glossary.jsonl"
DEFAULT_SHORTCODE_LUA = ROOT / "docs/_extensions/aria-glossary/glossary_terms.generated.lua"

REQUIRED_FIELDS = {
    "id",
    "anchor",
    "label",
    "category",
    "definition_short",
    "internal_links",
    "citations",
    "related",
    "kg_tags",
}


class GlossaryError(ValueError):
    """Raised when the glossary source is invalid."""


def _load_terms(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise GlossaryError(f"{path} must contain a YAML list of term records")
    terms: list[dict[str, Any]] = []
    for idx, raw in enumerate(data, start=1):
        if not isinstance(raw, dict):
            raise GlossaryError(f"term #{idx} must be a mapping")
        terms.append(raw)
    return terms


def _validate_terms(terms: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    anchors: set[str] = set()
    for term in terms:
        missing = sorted(REQUIRED_FIELDS - set(term))
        if missing:
            raise GlossaryError(f"{term.get('id', '<missing id>')} missing fields: {missing}")
        term_id = _expect_string(term, "id")
        anchor = _expect_string(term, "anchor")
        if term_id in ids:
            raise GlossaryError(f"duplicate term id: {term_id}")
        if anchor in anchors:
            raise GlossaryError(f"duplicate anchor: {anchor}")
        ids.add(term_id)
        anchors.add(anchor)
        if not anchor.startswith("term-"):
            raise GlossaryError(f"{term_id}: anchor must start with 'term-'")
        for field in ("aliases", "internal_links", "citations", "related", "kg_tags"):
            if field in term and not isinstance(term[field], list):
                raise GlossaryError(f"{term_id}: {field} must be a list")


def _expect_string(term: dict[str, Any], field: str) -> str:
    value = term.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GlossaryError(f"{term.get('id', '<missing id>')}: {field} must be a non-empty string")
    return value.strip()


def _as_list(term: dict[str, Any], field: str) -> list[str]:
    values = term.get(field) or []
    return [str(value) for value in values]


def _html_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _html_text(value: str) -> str:
    return html.escape(value, quote=False)


def _qmd_link(path: str, label: str | None = None, *, css_class: str = "glossary-chip") -> str:
    target = _qmd_target(path)
    text = label or _link_label(path)
    return f'<a class="{css_class}" href="{_html_attr(target)}" title="{_html_attr(path)}">{_html_text(text)}</a>'


def _qmd_target(path: str) -> str:
    anchor = ""
    if "#" in path:
        path, anchor_value = path.split("#", 1)
        anchor = "#" + anchor_value
    if path.startswith("docs/contents/"):
        return _rendered_doc_path(path.removeprefix("docs/contents/")) + anchor
    elif path.startswith("docs/reference/"):
        return "../reference/" + _rendered_doc_path(path.removeprefix("docs/reference/")) + anchor
    elif path.startswith("docs/"):
        return "../" + _rendered_doc_path(path.removeprefix("docs/")) + anchor
    return _rendered_doc_path(path) + anchor


def _rendered_doc_path(path: str) -> str:
    if path.endswith(".qmd"):
        return path.removesuffix(".qmd") + ".html"
    return path


def _link_label(path: str) -> str:
    path_without_anchor = path.split("#", 1)[0]
    if path_without_anchor.startswith("docs/reference/"):
        return Path(path_without_anchor).stem.split(".")[-1]
    target = ROOT / path_without_anchor
    if target.suffix == ".qmd" and target.is_file():
        title = _read_qmd_title(target)
        if title:
            return title
    if path_without_anchor.startswith("docs/typst/"):
        return Path(path_without_anchor).stem.replace("-", " ").replace("_", " ").title()
    stem = Path(path_without_anchor).stem
    return stem.replace("-", " ").replace("_", " ").title()


def _read_qmd_title(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front_matter = text[3:end]
            for line in front_matter.splitlines():
                if line.strip().startswith("title:"):
                    raw = line.split(":", 1)[1].strip().strip('"').strip("'")
                    raw = re.sub(r"\s*\{\s*#[^}]+\}\s*$", "", raw).strip()
                    if raw:
                        return raw
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)(?:\s+\{#.+\})?$", line.strip())
        if match:
            return match.group(1).strip("`")
    return None


def _term_title(term: dict[str, Any]) -> str:
    short = term.get("short")
    label = _expect_string(term, "label")
    if isinstance(short, str) and short and short != label:
        return f"{short} - {label}"
    return label


def _render_qmd(terms: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for term in sorted(terms, key=lambda item: (item["category"], item["label"].lower())):
        grouped.setdefault(term["category"].split(".", 1)[0], []).append(term)

    term_by_id = {term["id"]: term for term in terms}
    category_links = [
        f'<a class="glossary-chip glossary-category-chip" href="#glossary-category-{_slug(group)}">'
        f"{_html_text(_category_label(group))} <span>{len(group_terms)}</span></a>"
        for group, group_terms in grouped.items()
    ]
    lines = [
        "---",
        'title: "Glossary"',
        "format:",
        "  html:",
        "    code-tools: false",
        "bibliography: ../references.bib",
        "number-sections: false",
        "execute:",
        "  freeze: false",
        "---",
        "",
        "<!-- Generated by scripts/glossary_build.py; edit docs/glossary/terms.yml. -->",
        "",
        "::: {.glossary-page}",
        "",
        "::: {.glossary-intro}",
        f"**{len(terms)} terms** across **{len(grouped)} categories**. The canonical source is",
        "`docs/glossary/terms.yml`; generated Quarto, Typst, KG, and shortcode artifacts",
        "share the same definitions.",
        "",
        "Use `{{{< gls term-id >}}}` for the linked short label and",
        "`{{{< glsfull term-id >}}}` for the linked full label in QMD pages.",
        ":::",
        "",
        '<nav class="glossary-category-index" aria-label="Glossary categories">',
        *category_links,
        "</nav>",
        "",
    ]
    for group, group_terms in grouped.items():
        lines += [
            f"## {_category_label(group)} {{#glossary-category-{_slug(group)}}}",
            "",
            '<div class="glossary-card-grid">',
            "",
        ]
        for term in group_terms:
            lines += [
                f'<article class="glossary-card" id="{_html_attr(term["anchor"])}">',
                '<div class="glossary-card-header">',
                f'<div class="glossary-term-title">{_html_text(term["label"])}</div>',
                '<div class="glossary-term-badges">',
            ]
            short = term.get("short")
            if isinstance(short, str) and short and short != term["label"]:
                lines.append(f'<span class="glossary-badge glossary-badge-short">{_html_text(short)}</span>')
            lines += [
                f'<span class="glossary-badge">{_html_text(term["category"])}</span>',
                "</div>",
                "</div>",
                "",
            ]
            aliases = _as_list(term, "aliases")
            if aliases:
                lines += [
                    '<div class="glossary-aliases">',
                    "<span>Aliases</span>",
                    " ".join(f"<code>{_html_text(alias)}</code>" for alias in aliases),
                    "</div>",
                    "",
                ]
            lines += [f'<p class="glossary-definition">{_html_text(term["definition_short"].strip())}</p>', ""]
            if term.get("definition_long"):
                lines += [f'<p class="glossary-detail-text">{_html_text(str(term["definition_long"]).strip())}</p>', ""]
            formula = term.get("formula") or {}
            if isinstance(formula, dict) and formula.get("tex"):
                lines += [
                    '<div class="glossary-formula">',
                    r"\[",
                    _html_text(str(formula["tex"])),
                    r"\]",
                    "</div>",
                    "",
                ]
            lines += _render_metadata_details(term, term_by_id)
            lines += ["</article>", ""]
        lines += ["</div>", ""]
    all_citations = sorted({citation for term in terms for citation in _as_list(term, "citations")})
    if all_citations:
        lines += [
            "::: {.glossary-citation-seed}",
            "; ".join(f"[@{citation}]" for citation in all_citations),
            ":::",
            "",
        ]
    lines += [":::", ""]
    _write_text(path, "\n".join(lines).rstrip() + "\n")


def _render_metadata_details(term: dict[str, Any], term_by_id: dict[str, dict[str, Any]]) -> list[str]:
    related = []
    for related_id in _as_list(term, "related"):
        related_term = term_by_id.get(related_id)
        if related_term:
            related.append(
                f'<a class="glossary-chip" href="#{_html_attr(related_term["anchor"])}">'
                f"{_html_text(str(related_term.get('short') or related_term['label']))}</a>"
            )
        else:
            related.append(
                f'<a class="glossary-chip" href="#term-{_html_attr(_slug(related_id))}">{_html_text(related_id)}</a>'
            )
    docs = [_qmd_link(link, css_class="glossary-chip") for link in _as_list(term, "internal_links")]
    citations = [
        f'<a class="glossary-chip glossary-citation" href="#ref-{_html_attr(citation)}">&#64;{_html_text(citation)}</a>'
        for citation in _as_list(term, "citations")
    ]
    if not (related or docs or citations):
        return []

    lines = [
        '<details class="glossary-details">',
        "<summary>Links and references</summary>",
        '<div class="glossary-metadata">',
    ]
    if related:
        lines.append(f'<div class="glossary-metadata-row"><span>Related</span><div>{" ".join(related)}</div></div>')
    if docs:
        lines.append(f'<div class="glossary-metadata-row"><span>Docs</span><div>{" ".join(docs)}</div></div>')
    if citations:
        lines.append(
            f'<div class="glossary-metadata-row"><span>References</span><div>{" ".join(citations)}</div></div>'
        )
    lines += ["</div>", "</details>", ""]
    return lines


def _category_label(group: str) -> str:
    return group.replace("-", " ").replace("_", " ").title()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def _typst_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lua_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _typst_content(value: str) -> str:
    return "[" + value.replace("]", "\\]") + "]"


def _render_typst(terms: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "// Generated by scripts/glossary_build.py; edit docs/glossary/terms.yml.",
        "",
        "#let glossary = (",
    ]
    for term in sorted(terms, key=lambda item: item["id"]):
        short = term.get("short") or term["label"]
        lines.append(
            "  (id: "
            + _typst_string(term["id"])
            + ", label: "
            + _typst_string(term["label"])
            + ", short: "
            + _typst_string(str(short))
            + ", anchor: "
            + _typst_string(term["anchor"])
            + "),"
        )
    lines += [
        ")",
        "",
        "#let gls(id) = {",
    ]
    for index, term in enumerate(sorted(terms, key=lambda item: item["id"])):
        keyword = "if" if index == 0 else "} else if"
        lines.append(
            f"  {keyword} id == {_typst_string(term['id'])} {{ {_typst_content(str(term.get('short') or term['label']))} "
        )
    lines += [
        "  } else { id }",
        "}",
        "",
        "#let gls-full(id) = {",
    ]
    for index, term in enumerate(sorted(terms, key=lambda item: item["id"])):
        keyword = "if" if index == 0 else "} else if"
        short = term.get("short")
        full = term["label"] if not short or short == term["label"] else f"{term['label']} ({short})"
        lines.append(f"  {keyword} id == {_typst_string(term['id'])} {{ {_typst_content(full)} ")
    lines += [
        "  } else { id }",
        "}",
        "",
        "#let glossary-list() = [",
    ]
    for term in sorted(terms, key=lambda item: item["label"].lower()):
        definition = str(term["definition_short"]).strip()
        lines.append(f"  - *{_term_title(term)}*: {definition}")
    lines += ["]", ""]
    _write_text(path, "\n".join(lines))


def _render_jsonl(terms: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for term in sorted(terms, key=lambda item: item["id"]):
        rows.append(
            {
                "node_type": "Concept",
                "id": term["id"],
                "anchor": term["anchor"],
                "label": term["label"],
                "short": term.get("short"),
                "aliases": _as_list(term, "aliases"),
                "category": term["category"],
                "parent": term.get("parent"),
                "definition_short": str(term["definition_short"]).strip(),
                "definition_long": str(term.get("definition_long") or "").strip(),
                "internal_links": _as_list(term, "internal_links"),
                "citations": _as_list(term, "citations"),
                "related": _as_list(term, "related"),
                "kg_tags": _as_list(term, "kg_tags"),
                "notation": term.get("notation") or {},
                "formula": term.get("formula") or {},
            }
        )
    _write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _render_shortcode_lua(terms: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "-- Generated by scripts/glossary_build.py; edit docs/glossary/terms.yml.",
        "",
        "return {",
    ]
    for term in sorted(terms, key=lambda item: item["id"]):
        short = str(term.get("short") or term["label"])
        lines += [
            f"  [{_lua_string(term['id'])}] = {{",
            f"    label = {_lua_string(term['label'])},",
            f"    short = {_lua_string(short)},",
            f"    anchor = {_lua_string(term['anchor'])},",
            "  },",
        ]
    lines += ["}", ""]
    _write_text(path, "\n".join(lines))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    terms = _load_terms(args.terms)
    _validate_terms(terms)
    actions = set(args.actions)
    if "all" in actions:
        actions = {"validate", "render-quarto", "render-typst", "render-jsonl", "render-shortcode-lua"}
    if "render-quarto" in actions:
        _render_qmd(terms, args.qmd_out)
    if "render-typst" in actions:
        _render_typst(terms, args.typst_out)
    if "render-jsonl" in actions:
        _render_jsonl(terms, args.jsonl_out)
    if "render-shortcode-lua" in actions:
        _render_shortcode_lua(terms, args.shortcode_lua_out)
    if "validate" in actions:
        print(f"Validated {len(terms)} glossary terms from {args.terms}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "actions",
        nargs="*",
        default=["all"],
        choices=["all", "validate", "render-quarto", "render-typst", "render-jsonl", "render-shortcode-lua"],
    )
    parser.add_argument("--terms", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--qmd-out", type=Path, default=DEFAULT_QMD)
    parser.add_argument("--typst-out", type=Path, default=DEFAULT_TYPST)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--shortcode-lua-out", type=Path, default=DEFAULT_SHORTCODE_LUA)
    args = parser.parse_args()
    try:
        build(args)
    except GlossaryError as exc:
        raise SystemExit(f"glossary error: {exc}") from exc


if __name__ == "__main__":
    main()

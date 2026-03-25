#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"

OUT="${1:-${ROOT_DIR}/docs/_generated/context/source_index.md}"
if [[ "$OUT" != /* ]]; then
  OUT="${ROOT_DIR}/${OUT}"
fi

mkdir -p "$(dirname "$OUT")"

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
tmp="$(mktemp)"

relpath() {
  sed "s#^${ROOT_DIR}/##"
}

has_rg() {
  command -v rg >/dev/null 2>&1
}

list_files() {
  local pattern="$1"
  local root="$2"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  if has_rg; then
    rg --files -g "$pattern" "$root" 2>/dev/null || true
  else
    find "$root" -type f -name "$pattern" 2>/dev/null || true
  fi
}

count_files() {
  local pattern="$1"
  local root="$2"
  list_files "$pattern" "$root" | wc -l | tr -d ' '
}

qmd_count="$(count_files '*.qmd' "${ROOT_DIR}/docs")"
typst_count="$(count_files '*.typ' "${ROOT_DIR}/docs/typst")"
lit_tex_count="$(count_files '*.tex' "${ROOT_DIR}/literature")"
lit_bib_count="$(count_files '*.bib' "${ROOT_DIR}/literature")"
py_count="$(count_files '*.py' "${ROOT_DIR}/oracle_rri/oracle_rri")"
ref_count="$(count_files '*.md' "${ROOT_DIR}/.agents/references")"
memory_state_count="$(count_files '*.md' "${ROOT_DIR}/.agents/memory/state")"
memory_history_count="$(count_files '*.md' "${ROOT_DIR}/.agents/memory/history")"

{
  echo "# Context Sources Index"
  echo
  echo "- Generated: ${timestamp}"
  echo "- Repo: ${ROOT_DIR}"
  echo
  echo "## Retrieval ladder"
  echo "1. Fixed ground truth: docs/typst/paper/main.typ"
  echo "2. Canonical current truth: .agents/memory/state/{PROJECT_STATE,DECISIONS,OPEN_QUESTIONS,GOTCHAS}.md"
  echo "3. Agent references: .agents/references/{python_conventions,context7_library_ids}.md"
  echo "4. Checked-in routing map: .agents/skills/aria-nbv-context/references/context_map.md"
  echo "5. Source-specific reveal tools: qmd outline, typst includes, literature index, code AST summaries"
  echo "6. Heavyweight fallback: make context"
  echo
  echo "## Summary"
  echo "- Agent memory: ${memory_state_count} state docs, ${memory_history_count} history docs"
  echo "- Agent references: ${ref_count} files"
  echo "- Quarto docs: ${qmd_count} files"
  echo "- Typst (paper/slides/shared): ${typst_count} files"
  echo "- Literature: ${lit_tex_count} .tex, ${lit_bib_count} .bib"
  echo "- Python source: ${py_count} files"
  echo
  echo "## Fixed entrypoints"
  echo "- docs/typst/paper/main.typ"
  echo "- .agents/memory/state/PROJECT_STATE.md"
  echo "- .agents/memory/state/DECISIONS.md"
  echo "- .agents/memory/state/OPEN_QUESTIONS.md"
  echo "- .agents/memory/state/GOTCHAS.md"
  echo "- .agents/references/python_conventions.md"
  echo "- .agents/references/context7_library_ids.md"
  echo "- .agents/skills/aria-nbv-context/references/context_map.md"
  echo "- docs/_generated/context/data_contracts.md"
  echo
  echo "## Recommended reveal commands"
  echo '- scripts/nbv_qmd_outline.sh --compact'
  echo '- scripts/nbv_typst_includes.py --paper --mode outline'
  echo '- scripts/nbv_literature_index.sh'
  echo '- scripts/nbv_get_context.sh contracts'
  echo '- scripts/nbv_get_context.sh modules'
  echo '- scripts/nbv_get_context.sh match <term>'
  echo '- make context    # only when lighter tools are insufficient'
  echo
  echo "## Agent memory"
  echo "State docs:"
  list_files '*.md' "${ROOT_DIR}/.agents/memory/state" | sort | relpath || true
  echo
  echo "History:"
  echo "- Root: .agents/memory/history"
  if [[ -d "${ROOT_DIR}/.agents/memory/history" ]]; then
    find "${ROOT_DIR}/.agents/memory/history" -mindepth 1 -maxdepth 2 -type d | sort | relpath || true
  fi
  echo
  echo "## Agent references (.agents/references/*.md)"
  list_files '*.md' "${ROOT_DIR}/.agents/references" | sort | relpath || true
  echo
  echo "## Quarto docs (docs/**/*.qmd)"
  echo "Entry points: docs/index.qmd, docs/contents/todos.qmd, docs/contents/questions.qmd"
  list_files '*.qmd' "${ROOT_DIR}/docs" | sort | relpath || true
  echo
  echo "## Typst paper (docs/typst/paper/**/*.typ)"
  echo "Entry points: docs/typst/paper/main.typ"
  list_files '*.typ' "${ROOT_DIR}/docs/typst/paper" | sort | relpath || true
  echo
  echo "## Typst slides (docs/typst/slides/**/*.typ)"
  echo "Use only when the task explicitly touches slides."
  list_files '*.typ' "${ROOT_DIR}/docs/typst/slides" | sort | relpath || true
  echo
  echo "## Typst shared (docs/typst/shared/**/*.typ)"
  list_files '*.typ' "${ROOT_DIR}/docs/typst/shared" | sort | relpath || true
  echo
  echo "## Literature source families"
  if [[ -d "${ROOT_DIR}/literature/tex-src" ]]; then
    find "${ROOT_DIR}/literature/tex-src" -mindepth 1 -maxdepth 1 -type d | sort | relpath || true
  fi
  echo
  echo "## Literature sources (literature/**/*.tex, literature/**/*.bib)"
  list_files '*.tex' "${ROOT_DIR}/literature" | sort | relpath || true
  list_files '*.bib' "${ROOT_DIR}/literature" | sort | relpath || true
  echo
  echo "## Python source (oracle_rri/**)"
  echo "Entry points: oracle_rri/AGENTS.md, docs/_generated/context/data_contracts.md, oracle_rri/oracle_rri/data, oracle_rri/oracle_rri/pipelines, oracle_rri/oracle_rri/vin"
  list_files '*.py' "${ROOT_DIR}/oracle_rri/oracle_rri" | sort | relpath || true
  echo
  echo "## Search recipes (rg)"
  echo 'rg -n "<term>" .agents/memory/state .agents/memory/history'
  echo 'rg -n "<term>" .agents/references'
  echo 'rg -n "<term>" docs/**/*.qmd'
  echo 'rg -n "<term>" docs/typst/**/*.typ'
  echo 'rg -n "<term>" literature/**/*.{tex,bib,sty}'
  echo 'rg -n "<term>" oracle_rri/oracle_rri'
  echo 'rg -n "VinPrediction|EfmSnippetView|BaseConfig" docs/_generated/context/data_contracts.md'
} > "$tmp"

mv "$tmp" "$OUT"
echo "Wrote context sources index to $OUT"

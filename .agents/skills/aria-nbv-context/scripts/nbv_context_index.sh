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

list_files() {
  local pattern="$1"
  local root="$2"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -type f -name "$pattern" 2>/dev/null || true
}

count_files() {
  local pattern="$1"
  local root="$2"
  list_files "$pattern" "$root" | wc -l | tr -d ' '
}

count_immediate_dirs() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  find "$root" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '
}

qmd_count="$(count_files '*.qmd' "${ROOT_DIR}/docs")"
typst_paper_count="$(count_files '*.typ' "${ROOT_DIR}/docs/typst/paper")"
typst_slides_count="$(count_files '*.typ' "${ROOT_DIR}/docs/typst/slides")"
typst_shared_count="$(count_files '*.typ' "${ROOT_DIR}/docs/typst/shared")"
lit_bib_count="$(count_files '*.bib' "${ROOT_DIR}/docs/literature")"
lit_family_count="$(count_immediate_dirs "${ROOT_DIR}/docs/literature/tex-src")"
lit_tex_count="$(count_files '*.tex' "${ROOT_DIR}/docs/literature")"
py_count="$(count_files '*.py' "${ROOT_DIR}/aria_nbv/aria_nbv")"
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
  echo "2. Default current-state read: .agents/memory/state/PROJECT_STATE.md"
  echo "3. Durable owner directives: .agents/memory/state/OWNER_DIRECTIVES.md"
  echo "4. Compact routing index: docs/_generated/context/source_index.md"
  echo '5. Deepest relevant path-local `AGENTS.md` once the task is localized'
  echo "6. On-demand state docs: .agents/memory/state/{DECISIONS,OPEN_QUESTIONS,GOTCHAS}.md"
  echo "7. On-demand references: .agents/references/{operator_quick_reference,python_conventions,agent_memory_templates,context7_library_ids,tooling_skill_governance}.md"
  echo "8. Checked-in routing map: .agents/skills/aria-nbv-context/references/context_map.md"
  echo "9. Lightweight refresh: make context"
  echo "10. Heavyweight fallback: make context-heavy"
  echo
  echo "## Hot-path bundle"
  echo "- docs/typst/paper/main.typ"
  echo "- .agents/memory/state/PROJECT_STATE.md"
  echo "- .agents/memory/state/OWNER_DIRECTIVES.md"
  echo "- docs/_generated/context/source_index.md"
  echo
  echo "## Path-local boundary guides"
  echo "- aria_nbv/AGENTS.md"
  echo "- aria_nbv/aria_nbv/vin/AGENTS.md"
  echo "- aria_nbv/aria_nbv/data_handling/AGENTS.md"
  echo "- aria_nbv/aria_nbv/rri_metrics/AGENTS.md"
  echo "- docs/AGENTS.md"
  echo "- docs/typst/paper/AGENTS.md"
  echo
  echo "## Lightweight refresh"
  echo "- \`make context\` refreshes \`source_index.md\`, \`literature_index.md\`, and \`data_contracts.md\`."
  echo "- Prefer \`make context-contracts\` or \`scripts/nbv_get_context.sh contracts\` before opening \`docs/_generated/context/data_contracts.md\`."
  echo "- Use \`make context-heavy\` only for bundled heavy fallback artifacts."
  echo
  echo "## Source families"
  echo "| Family | Count | Use when | First reveal |"
  echo "|---|---:|---|---|"
  echo "| Canonical state | ${memory_state_count} docs | You need current truth, conventions, or decisions | Open the relevant doc in \`.agents/memory/state/\` |"
  echo "| Agent history | ${memory_history_count} docs | Historical evidence only; open only after current state docs are insufficient | \`rg -n \"<term>\" .agents/memory/history\` |"
  echo "| Agent references | ${ref_count} docs | You need conventions, templates, or external-doc lookup ids | Open \`python_conventions.md\` or the specific reference doc |"
  echo "| Quarto docs | ${qmd_count} files | You need implementation narrative, roadmap, or explainer docs | \`scripts/nbv_qmd_outline.sh --compact\` |"
  echo "| Typst paper | ${typst_paper_count} files | You need the authoritative research narrative | \`scripts/nbv_typst_includes.py --paper --mode outline\` |"
  echo "| Typst slides/shared | $((typst_slides_count + typst_shared_count)) files | The task explicitly touches slides or shared macros | \`scripts/nbv_typst_includes.py --with-slides --mode outline\` |"
  echo "| Literature | ${lit_family_count} families, ${lit_tex_count} .tex, ${lit_bib_count} .bib | You need related-work or source-paper context | \`scripts/nbv_literature_index.sh\` |"
  echo "| Python source | ${py_count} files | You need contracts, module ownership, or symbols | \`scripts/nbv_get_context.sh contracts\`, \`modules\`, or \`match <term>\` |"
  echo
  echo "## Curated documentation families"
  echo "| Topic | Primary paths | Use when |"
  echo "|---|---|---|"
  echo "| Project hub | \`docs/index.qmd\` | You need the docs landing page or high-level navigation. |"
  echo "| Active work | \`docs/contents/todos.qmd\`, \`roadmap.qmd\`, \`questions.qmd\` | You need current work items, milestones, or open research questions. |"
  echo "| Setup and resources | \`docs/contents/setup.qmd\`, \`resources.qmd\` | You need environment/bootstrap help or external resource links. |"
  echo "| Findings and glossary | \`docs/contents/experiments/findings.qmd\`, \`glossary.qmd\` | You need prior experiment outcomes or project terminology. |"
  echo "| Internal implementation | \`docs/contents/impl/overview.qmd\`, \`rri_computation.qmd\` | You need package architecture or oracle/RRI computation details. |"
  echo "| External stack notes | \`docs/contents/ext-impl/efm3d_symbol_index.qmd\`, \`prj_aria_tools_impl.qmd\` | You need vendor symbol maps or Project Aria tooling details. |"
  echo
  echo "## Secondary/on-demand references"
  echo "- .agents/references/operator_quick_reference.md"
  echo "- .agents/references/python_conventions.md"
  echo "- .agents/skills/aria-nbv-context/references/context_map.md"
  echo "- .agents/references/agent_memory_templates.md"
  echo "- .agents/references/context7_library_ids.md"
  echo "- .agents/references/tooling_skill_governance.md"
  echo "- .agents/memory/state/DECISIONS.md"
  echo "- .agents/memory/state/OPEN_QUESTIONS.md"
  echo "- .agents/memory/state/GOTCHAS.md"
  echo
  echo "## Preferred reveal commands"
  echo '- scripts/nbv_qmd_outline.sh --compact'
  echo '- scripts/nbv_typst_includes.py --paper --mode outline'
  echo '- scripts/nbv_literature_index.sh'
  echo '- make context-contracts'
  echo '- scripts/nbv_get_context.sh contracts'
  echo '- scripts/nbv_get_context.sh modules'
  echo '- scripts/nbv_get_context.sh match <term>'
  echo '- make context    # refresh lightweight routing artifacts'
  echo
  echo "## Heavy fallback only"
  echo "- Prefer specific targets when possible: \`make context-uml\`, \`make context-docstrings\`, \`make context-tree\`."
  echo "- Bundled fallback: \`make context-heavy\`."
  echo "- Heavy artifacts: \`context_snapshot.md\`, \`aria_nbv_uml.mmd\`, \`aria_nbv_filtered_uml.mmd\`, \`aria_nbv_class_docstrings.md\`, \`aria_nbv_tree.md\`."
  echo
  echo "## Search recipes (rg)"
  echo 'rg -n "<term>" .agents/memory/state'
  echo 'rg -n "<term>" .agents/memory/history'
  echo 'rg -n "<term>" .agents/references'
  echo 'rg -n "<term>" docs/**/*.qmd'
  echo 'rg -n "<term>" docs/typst/**/*.typ'
  echo 'rg -n "<term>" docs/literature/**/*.{tex,bib,sty}'
  echo 'rg -n "<term>" aria_nbv/aria_nbv'
  echo 'rg -n "VinPrediction|EfmSnippetView|BaseConfig" docs/_generated/context/data_contracts.md'
} > "$tmp"

mv "$tmp" "$OUT"
echo "Wrote context sources index to $OUT"

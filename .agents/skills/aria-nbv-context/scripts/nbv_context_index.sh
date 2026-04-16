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
lit_tex_count="$(count_files '*.tex' "${ROOT_DIR}/literature")"
lit_bib_count="$(count_files '*.bib' "${ROOT_DIR}/literature")"
lit_family_count="$(count_immediate_dirs "${ROOT_DIR}/literature/tex-src")"
py_count="$(count_files '*.py' "${ROOT_DIR}/aria_nbv/aria_nbv")"
ref_count="$(count_files '*.md' "${ROOT_DIR}/.agents/references")"
skill_count="$(count_files 'SKILL.md' "${ROOT_DIR}/.agents/skills")"
memory_state_count="$(count_files '*.md' "${ROOT_DIR}/.agents/memory/state")"
memory_history_count="$(count_files '*.md' "${ROOT_DIR}/.agents/memory/history")"
local_agents_count="$(
  find "${ROOT_DIR}" \
    -path "${ROOT_DIR}/.git" -prune -o \
    -path "${ROOT_DIR}/.agents/archive" -prune -o \
    -path "${ROOT_DIR}/aria_nbv/.venv" -prune -o \
    -name AGENTS.md -type f -print \
    | wc -l \
    | tr -d ' '
)"

{
  echo "# Context Sources Index"
  echo
  echo "- Generated: ${timestamp}"
  echo "- Repo: ${ROOT_DIR}"
  echo
  echo "## Retrieval ladder"
  echo "1. Fixed ground truth: docs/typst/paper/main.typ"
  echo "2. Canonical current truth: .agents/memory/state/{PROJECT_STATE,DECISIONS,OPEN_QUESTIONS,GOTCHAS}.md"
  echo "3. Compact routing index: docs/_generated/context/source_index.md"
  echo "4. Deepest relevant path-local AGENTS.md once the task is localized"
  echo "5. On-demand references: .agents/references/{operator_quick_reference,python_conventions,agent_memory_templates,context7_library_ids}.md"
  echo "6. Narrow skills: aria-nbv-docs-context, aria-nbv-code-context, aria-nbv-scaffold-maintenance"
  echo "7. Checked-in routing map: .agents/skills/aria-nbv-context/references/context_map.md"
  echo "8. Startup hook refreshes the lightweight context bundle for new trusted Codex sessions"
  echo "9. Manual refresh after scaffold or routing edits: make context"
  echo "10. Heavyweight fallback: make context-heavy"
  echo
  echo "## Hot-path bundle"
  echo "- docs/typst/paper/main.typ"
  echo "- .agents/memory/state/PROJECT_STATE.md"
  echo "- .agents/memory/state/DECISIONS.md"
  echo "- .agents/memory/state/OPEN_QUESTIONS.md"
  echo "- .agents/memory/state/GOTCHAS.md"
  echo "- docs/_generated/context/source_index.md"
  echo "- docs/_generated/context/literature_index.md"
  echo "- docs/_generated/context/data_contracts.md"
  echo
  echo "## Path-local boundary guides"
  find "${ROOT_DIR}" \
    -path "${ROOT_DIR}/.git" -prune -o \
    -path "${ROOT_DIR}/.agents/archive" -prune -o \
    -path "${ROOT_DIR}/aria_nbv/.venv" -prune -o \
    -name AGENTS.md -type f -print \
    | relpath \
    | sort \
    | sed 's#^#- #'
  echo
  echo "## Lightweight refresh"
  echo "- New trusted Codex sessions normally start with hook-refreshed \`source_index.md\`, \`literature_index.md\`, and \`data_contracts.md\`."
  echo "- Run \`make context\` manually after routing or scaffold edits, when hooks are disabled, or when generated context appears stale."
  echo "- Prefer \`make context-contracts\` or \`scripts/nbv_get_context.sh contracts\` before opening \`docs/_generated/context/data_contracts.md\`."
  echo "- Use \`make context-heavy\` only for bundled heavy fallback artifacts."
  echo
  echo "## Source families"
  echo "| Family | Count | Use when | First reveal |"
  echo "|---|---:|---|---|"
  echo "| Canonical state | ${memory_state_count} docs | You need current truth, conventions, or decisions | Open the relevant doc in \`.agents/memory/state/\` |"
  echo "| Agent history | ${memory_history_count} docs | The task is historical, comparative, or evidence-driven | \`rg -n \"<term>\" .agents/memory/history\` |"
  echo "| Agent references | ${ref_count} docs | You need conventions, templates, or external-doc lookup ids | Open \`python_conventions.md\` or the specific reference doc |"
  echo "| Repo skills | ${skill_count} skills | You need a workflow beyond nearest AGENTS guidance | Open the relevant \`.agents/skills/*/SKILL.md\` |"
  echo "| Path-local AGENTS | ${local_agents_count} files | You are editing a localized subtree | Open the deepest matching \`AGENTS.md\` |"
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
  echo "- .agents/references/gitnexus_optional.md"
  echo "- .agents/references/codex_hooks.md"
  echo "- .agents/AGENTS_INTERNAL_DB.md"
  echo "- .agents/issues.toml"
  echo "- .agents/todos.toml"
  echo "- docs/index.qmd"
  echo "- docs/contents/todos.qmd"
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
  echo 'rg -n "<term>" literature/**/*.{tex,bib,sty}'
  echo 'rg -n "<term>" aria_nbv/aria_nbv'
  echo 'rg -n "VinPrediction|EfmSnippetView|BaseConfig" docs/_generated/context/data_contracts.md'
} > "$tmp"

mv "$tmp" "$OUT"
echo "Wrote context sources index to $OUT"

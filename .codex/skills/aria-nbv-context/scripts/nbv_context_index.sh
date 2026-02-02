#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"

OUT="${1:-${ROOT_DIR}/.codex/context_sources_index.md}"
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

{
  echo "# Context Sources Index"
  echo
  echo "- Generated: ${timestamp}"
  echo "- Repo: ${ROOT_DIR}"
  echo
  echo "## Summary"
  echo "- Quarto docs: ${qmd_count} files"
  echo "- Typst (paper/slides/shared): ${typst_count} files"
  echo "- Literature: ${lit_tex_count} .tex, ${lit_bib_count} .bib"
  echo "- Python source: ${py_count} files"
  echo
  echo "## Quarto docs (docs/**/*.qmd)"
  list_files '*.qmd' "${ROOT_DIR}/docs" | sort | relpath || true
  echo
  echo "## Typst paper (docs/typst/paper/**/*.typ)"
  list_files '*.typ' "${ROOT_DIR}/docs/typst/paper" | sort | relpath || true
  echo
  echo "## Typst slides (docs/typst/slides/**/*.typ)"
  list_files '*.typ' "${ROOT_DIR}/docs/typst/slides" | sort | relpath || true
  echo
  echo "## Typst shared (docs/typst/shared/**/*.typ)"
  list_files '*.typ' "${ROOT_DIR}/docs/typst/shared" | sort | relpath || true
  echo
  echo "## Literature sources (literature/**/*.tex, literature/**/*.bib)"
  list_files '*.tex' "${ROOT_DIR}/literature" | sort | relpath || true
  list_files '*.bib' "${ROOT_DIR}/literature" | sort | relpath || true
  echo
  echo "## Python source (oracle_rri/**)"
  list_files '*.py' "${ROOT_DIR}/oracle_rri/oracle_rri" | sort | relpath || true
  echo
  echo "## Search recipes (rg)"
  echo "rg -n \"<term>\" docs/**/*.qmd"
  echo "rg -n \"<term>\" docs/typst/**/*.typ"
  echo "rg -n \"<term>\" literature/**/*.{tex,bib}"
  echo "rg -n \"<term>\" oracle_rri/oracle_rri"
  echo "rg -n \"VIN-NBV\" literature/**/*.{tex,bib}"
} > "$tmp"

mv "$tmp" "$OUT"
echo "Wrote context sources index to $OUT"

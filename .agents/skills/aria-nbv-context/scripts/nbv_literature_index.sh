#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"
LIT_DIR="${ROOT_DIR}/docs/literature"
TEX_SRC_DIR="${LIT_DIR}/tex-src"
REGISTRY_PATH="${ROOT_DIR}/.agents/kg/generated/literature/registry.jsonl"
OUT="${1:-${ROOT_DIR}/docs/_generated/context/literature_index.md}"

if [[ "$OUT" != /* ]]; then
  OUT="${ROOT_DIR}/${OUT}"
fi

mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp)"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

relpath() {
  sed "s#^${ROOT_DIR}/##"
}

{
  echo "# Literature Source Index"
  echo
  echo "- Generated: ${timestamp}"
  echo "- Repo: ${ROOT_DIR}"
  echo
  echo "## How to use"
  echo "1. Pick the paper family here."
  echo "2. Run scripts/nbv_literature_search.sh \"<term>\" within that family or globally."
  echo "3. Open only the matching section files after the family is known."
  echo
  if [[ -f "$REGISTRY_PATH" ]]; then
    python3 - "$ROOT_DIR" "$REGISTRY_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
registry_path = Path(sys.argv[2])


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


records = [
    json.loads(line)
    for line in registry_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
records.sort(
    key=lambda record: (
        str(record.get("year") or "9999"),
        str(record.get("title") or record.get("paper_id") or "").lower(),
    )
)
enriched = [record for record in records if record.get("semantic_scholar")]
plain = [record for record in records if not record.get("semantic_scholar")]
local_tex = [record for record in records if record.get("has_local_tex")]
local_pdf = [record for record in records if record.get("has_local_pdf")]

print("## litkg Registry")
print(f"- Registry: `{rel(registry_path)}`")
print(f"- Records: {len(records)}")
print(f"- Semantic Scholar enriched: {len(enriched)}")
print(f"- Local TeX records: {len(local_tex)}")
print(f"- Local PDF records: {len(local_pdf)}")
print()

if enriched:
    print("### Semantic Scholar Enriched Records")
    print("| ID | Year | Citations | Title | Semantic Scholar |")
    print("|---|---:|---:|---|---|")
    for record in enriched:
        s2 = record["semantic_scholar"]
        s2_url = s2.get("url") or ""
        s2_label = s2.get("paperId") or s2.get("corpusId") or ""
        if s2_url and s2_label:
            s2_cell = f"[{s2_label}]({s2_url})"
        else:
            s2_cell = s2_label or s2_url
        print(
            "| "
            + " | ".join(
                [
                    cell(record.get("citation_key") or record.get("paper_id")),
                    cell(s2.get("year") or record.get("year")),
                    cell(s2.get("citationCount")),
                    cell(s2.get("title") or record.get("title")),
                    cell(s2_cell),
                ]
            )
            + " |"
        )
    print()

if plain:
    print("### Registry Records Without Semantic Scholar Metadata")
    print("| ID | Year | Source | Title |")
    print("|---|---:|---|---|")
    for record in plain:
        print(
            "| "
            + " | ".join(
                [
                    cell(record.get("citation_key") or record.get("paper_id")),
                    cell(record.get("year")),
                    cell(record.get("source_kind")),
                    cell(record.get("title")),
                ]
            )
            + " |"
        )
    print()
PY
  else
    echo "## litkg Registry"
    echo "- No litkg registry found at \`.agents/kg/generated/literature/registry.jsonl\`."
    echo "- Run \`make kg-sync\` and \`make kg-semantic-enrich\` before relying on Semantic Scholar metadata."
    echo
  fi
  if [[ -d "$TEX_SRC_DIR" ]]; then
    echo "## Local TeX Paper Families"
    while IFS= read -r family; do
      rel_family="$(printf '%s\n' "$family" | relpath)"
      family_name="$(basename "$family")"
      main_tex=""
      if [[ -f "$family/main.tex" ]]; then
        main_tex="$(printf '%s\n' "$family/main.tex" | relpath)"
      fi
      mapfile -t bibs < <(find "$family" -maxdepth 2 -type f -name '*.bib' | sort)
      mapfile -t texs < <(find "$family" -type f -name '*.tex' ! -name 'main.tex' | sort)
      echo "### ${family_name}"
      echo "- Root: ${rel_family}"
      if [[ -n "$main_tex" ]]; then
        echo "- Main: ${main_tex}"
      fi
      if [[ ${#bibs[@]} -gt 0 ]]; then
        echo "- Bibliography:"
        for bib in "${bibs[@]}"; do
          printf '  - %s\n' "$(printf '%s\n' "$bib" | relpath)"
        done
      fi
      if [[ ${#texs[@]} -gt 0 ]]; then
        echo "- Sections:"
        for tex in "${texs[@]}"; do
          printf '  - %s\n' "$(printf '%s\n' "$tex" | relpath)"
        done
      fi
      echo
    done < <(find "$TEX_SRC_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
  else
    echo "## Paper families"
    echo "- No literature/tex-src directory found."
  fi
  echo "## Search recipes"
  echo 'scripts/nbv_literature_search.sh "VIN-NBV"'
  echo 'scripts/nbv_literature_search.sh "Relative Reconstruction Improvement"'
  echo 'scripts/nbv_literature_search.sh "Aria Synthetic Environments"'
} > "$tmp"

mv "$tmp" "$OUT"
echo "Wrote literature index to $OUT"

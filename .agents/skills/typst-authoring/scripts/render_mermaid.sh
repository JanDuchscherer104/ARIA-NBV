#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: render_mermaid.sh -i <input.mmd> -o <output.{png,svg,pdf}> [options]

Options:
  -i, --input       Mermaid source file.
  -o, --out         Rendered output file.
      --theme       Mermaid theme (default: neutral).
      --background  Background color (default: transparent).
      --width       Viewport width in px (default: 1600).
      --height      Viewport height in px (optional).
      --pdf-fit     Pass --pdfFit to Mermaid CLI for PDF output.
  -h, --help        Show this help.

Example:
  .agents/skills/typst-authoring/scripts/render_mermaid.sh \
    -i docs/typst/thesis/figures/proposal_system_flow.mmd \
    -o docs/typst/thesis/figures/proposal_system_flow.png \
    --width 1600
EOF
}

input=""
output=""
theme="neutral"
background="transparent"
width="1600"
height=""
pdf_fit="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      input="$2"
      shift 2
      ;;
    -o|--out)
      output="$2"
      shift 2
      ;;
    --theme)
      theme="$2"
      shift 2
      ;;
    --background)
      background="$2"
      shift 2
      ;;
    --width)
      width="$2"
      shift 2
      ;;
    --height)
      height="$2"
      shift 2
      ;;
    --pdf-fit)
      pdf_fit="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$input" || -z "$output" ]]; then
  echo "Missing required --input or --out" >&2
  usage
  exit 1
fi

if ! command -v mmdc >/dev/null 2>&1; then
  echo "Mermaid CLI (mmdc) not found on PATH" >&2
  exit 127
fi

mkdir -p "$(dirname "$output")"

cmd=(mmdc -i "$input" -o "$output" -t "$theme" -b "$background" -w "$width")
if [[ -n "$height" ]]; then
  cmd+=(-H "$height")
fi
if [[ "$pdf_fit" == "true" ]]; then
  cmd+=(--pdfFit)
fi

printf 'Running:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2

"${cmd[@]}"

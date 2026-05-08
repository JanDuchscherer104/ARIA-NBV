#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: render_png.sh -i <input.typ> [-o <out_dir>] [-p <ppi>] [--pages <ranges>]

Options:
  -i, --input     Path to .typ file (required)
  -o, --out       Output directory (default: ./out)
  -p, --ppi       Pixels per inch (default: 300)
      --pages     Page ranges (e.g., "1", "2,3,7-9,11-")

Examples:
  render_png.sh -i figure.typ
  render_png.sh -i table.typ -o /tmp/renders --ppi 600 --pages 1
EOF
}

input=""
out_dir="./out"
ppi="300"
pages=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      input="$2"
      shift 2
      ;;
    -o|--out)
      out_dir="$2"
      shift 2
      ;;
    -p|--ppi)
      ppi="$2"
      shift 2
      ;;
    --pages)
      pages="$2"
      shift 2
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

if [[ -z "$input" ]]; then
  echo "Missing required --input" >&2
  usage
  exit 1
fi

mkdir -p "$out_dir"

output_template="${out_dir}/{0p}.png"

args=(compile "$input" "$output_template" --format png --ppi "$ppi")
if [[ -n "$pages" ]]; then
  args+=(--pages "$pages")
fi

typst "${args[@]}"

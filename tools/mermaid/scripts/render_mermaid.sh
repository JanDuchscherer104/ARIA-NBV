#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <diagram.mmd> [output.svg|output.png|output.pdf]" >&2
  exit 2
fi

input="$1"
output="${2:-${input%.mmd}.svg}"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "ERROR: mmdc not found." >&2
  echo "Install locally with: npm install -g @mermaid-js/mermaid-cli" >&2
  echo "For thesis figures, prefer local rendering; avoid online renderers unless explicitly permitted." >&2
  exit 127
fi

mmdc -i "$input" -o "$output" -b white -t default
printf 'Rendered %s -> %s\n' "$input" "$output"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: scripts/validate_mermaid.sh <input.mmd> [output.svg]" >&2
  exit 2
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.mmd}.svg}"
MMDC_BIN="${MMDC:-mmdc}"

if [[ ! -f "$INPUT" ]]; then
  echo "Input Mermaid file not found: $INPUT" >&2
  exit 1
fi

if ! command -v "$MMDC_BIN" >/dev/null 2>&1; then
  echo "Mermaid CLI not found: $MMDC_BIN" >&2
  echo "Install or point MMDC at a preinstalled local mermaid-cli binary." >&2
  exit 1
fi

"$MMDC_BIN" -i "$INPUT" -o "$OUTPUT"
echo "Validated Mermaid diagram: $OUTPUT"

#!/usr/bin/env bash
set -euo pipefail
HOOK_NAME="${1:?Usage: mempal-hook.sh <hook-name>}"
INPUT_FILE=$(mktemp) || { echo "Failed to create temp file" >&2; exit 1; }
cat > "$INPUT_FILE"
PYTHON_BIN="${CODEX_PLUGIN_ROOT}/../aria_nbv/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
if ! cat "$INPUT_FILE" | "$PYTHON_BIN" -m mempalace hook run --hook "$HOOK_NAME" --harness codex; then
  echo "MemPalace hook skipped or failed for ${HOOK_NAME}" >&2
fi
rm -f "$INPUT_FILE" 2>/dev/null
exit 0

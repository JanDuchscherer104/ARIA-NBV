#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "${ROOT_DIR}/oracle_rri/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/oracle_rri/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

exec "$PYTHON" "${SCRIPT_DIR}/nbv_qmd_outline.py" "$@"

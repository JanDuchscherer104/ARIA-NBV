#!/usr/bin/env bash
set -euo pipefail

mode="${1:-packages}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"

root="${2:-${ROOT_DIR}/oracle_rri/oracle_rri}"
if [[ "$root" != /* ]]; then
  root="${ROOT_DIR}/${root}"
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "${ROOT_DIR}/oracle_rri/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/oracle_rri/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

exec "$PYTHON" "${ROOT_DIR}/oracle_rri/scripts/get_context.py" "$mode" --root "$root"

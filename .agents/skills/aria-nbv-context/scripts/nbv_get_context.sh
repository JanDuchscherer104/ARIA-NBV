#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../" && pwd)"
DEFAULT_ROOT="${ROOT_DIR}/aria_nbv/aria_nbv"

mode="${1:-packages}"
shift || true

root_override=""
if [[ $# -gt 0 && "$1" != "--root" && "$mode" != "match" ]]; then
  case "$1" in
    /*|./*|../*|aria_nbv/*)
      root_override="$1"
      shift
      ;;
  esac
fi

args=("$mode")
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi

has_root=false
for arg in "${args[@]}"; do
  if [[ "$arg" == "--root" ]]; then
    has_root=true
    break
  fi
done

if [[ "$has_root" == false ]]; then
  root_path="${root_override:-${DEFAULT_ROOT}}"
  if [[ "$root_path" != /* ]]; then
    root_path="${ROOT_DIR}/${root_path}"
  fi
  args+=(--root "$root_path")
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "${ROOT_DIR}/aria_nbv/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/aria_nbv/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

exec "$PYTHON" "${ROOT_DIR}/aria_nbv/scripts/get_context.py" "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(git rev-parse --show-toplevel)
TOOLKIT_ROOT="${PROJECT_ROOT}/.agents/external/litkg-rs"
TARGET="${KG_SRC_DIR:-aria_nbv/aria_nbv}"

if [[ ! -x "${TOOLKIT_ROOT}/scripts/kg/index_code.sh" ]]; then
  echo "Missing litkg-rs code-index helper at ${TOOLKIT_ROOT}/scripts/kg/index_code.sh" >&2
  exit 2
fi

if [[ "$#" -gt 0 && "${1}" != -* ]]; then
  TARGET="$1"
  shift
fi
KG_CODE_REPO_ROOT="${PROJECT_ROOT}" \
  "${TOOLKIT_ROOT}/scripts/kg/index_code.sh" "${TARGET}" "$@"

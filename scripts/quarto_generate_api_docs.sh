#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOCS_DIR="${REPO_ROOT}/docs"
PYTHON_BIN=""
TMP_LOG="$(mktemp)"

cd "${DOCS_DIR}"
mkdir -p reference

cleanup() {
  rm -f "${TMP_LOG}"
}
trap cleanup EXIT

if [[ -n "${QUARTO_PYTHON:-}" && -x "${QUARTO_PYTHON}" ]]; then
  PYTHON_BIN="${QUARTO_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${REPO_ROOT}/aria_nbv/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/aria_nbv/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Could not find a Python executable for docs generation." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m quartodoc --help >/dev/null 2>&1; then
  if command -v uvx >/dev/null 2>&1; then
    echo "Using uvx fallback for quartodoc; local Python ${PYTHON_BIN} has no quartodoc."
    UVX_QUARTODOC="1"
  else
    echo "quartodoc is required to build docs/reference. Install quartodoc in ${PYTHON_BIN} or uvx." >&2
    exit 1
  fi
fi

set +e
if [[ "${UVX_QUARTODOC:-0}" == "1" ]]; then
  uvx --from quartodoc quartodoc build --config _quarto.yml 2>&1 | tee "${TMP_LOG}"
  BUILD_STATUS=${PIPESTATUS[0]}
else
  "${PYTHON_BIN}" -m quartodoc build --config _quarto.yml 2>&1 | tee "${TMP_LOG}"
  BUILD_STATUS=${PIPESTATUS[0]}
fi
set -e

if [[ "${BUILD_STATUS}" -ne 0 ]]; then
  echo "quartodoc build failed (exit ${BUILD_STATUS})." >&2
  exit "${BUILD_STATUS}"
fi

if grep -Eq "AliasResolutionError|KeyError: '" "${TMP_LOG}"; then
  echo "Detected hard API docs alias errors during Quartodoc build." >&2
  echo "Treat these as blockers and align docs/config exports before retrying." >&2
  exit 1
fi

if grep -Eq "^WARNING:" "${TMP_LOG}"; then
  echo "quartodoc finished with non-blocking warnings. Review them before release."
fi

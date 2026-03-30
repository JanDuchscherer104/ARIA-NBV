#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOCS_DIR="${REPO_ROOT}/docs"

cd "${DOCS_DIR}"
mkdir -p reference

if python3 -m quartodoc --help >/dev/null 2>&1; then
  python3 -m quartodoc build --config _quarto.yml
elif command -v uvx >/dev/null 2>&1; then
  uvx --from quartodoc quartodoc build --config _quarto.yml
else
  echo "quartodoc is required to build docs/reference. Install quartodoc or uv." >&2
  exit 1
fi

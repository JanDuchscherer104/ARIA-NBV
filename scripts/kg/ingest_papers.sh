#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(git rev-parse --show-toplevel)
CONFIG_PATH="${PROJECT_ROOT}/.configs/litkg.toml"
TOOLKIT_ROOT="${PROJECT_ROOT}/.agents/external/litkg-rs"
MODE="${1:-all}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ ! -f "${TOOLKIT_ROOT}/Cargo.toml" ]]; then
  echo "Missing litkg-rs submodule at ${TOOLKIT_ROOT}" >&2
  exit 2
fi

run_litkg() {
  cargo run --manifest-path "${TOOLKIT_ROOT}/Cargo.toml" -p litkg-cli -- "$@"
}

case "${MODE}" in
  sync)
    run_litkg sync-registry --config "${CONFIG_PATH}"
    ;;
  download)
    run_litkg download --config "${CONFIG_PATH}"
    ;;
  parse)
    run_litkg parse --config "${CONFIG_PATH}"
    ;;
  materialize)
    run_litkg materialize --config "${CONFIG_PATH}"
    ;;
  export-neo4j)
    run_litkg export-neo4j --config "${CONFIG_PATH}"
    ;;
  semantic-enrich)
    run_litkg enrich-semantic-scholar --config "${CONFIG_PATH}"
    ;;
  all)
    run_litkg sync-registry --config "${CONFIG_PATH}"
    run_litkg download --config "${CONFIG_PATH}"
    run_litkg parse --config "${CONFIG_PATH}"
    run_litkg materialize --config "${CONFIG_PATH}"
    run_litkg export-neo4j --config "${CONFIG_PATH}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: sync, download, parse, materialize, export-neo4j, semantic-enrich, all" >&2
    exit 2
    ;;
esac

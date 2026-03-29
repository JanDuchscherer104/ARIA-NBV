#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <oracle-cache-dir> <out-store-dir> [vin-cache-dir] [extra convert args...]" >&2
  exit 1
fi

ORACLE_CACHE="$1"
OUT_STORE="$2"
VIN_CACHE="${3:-}"

ROOT_DIR="/home/jandu/repos/NBV"
SCRIPT_DIR="$ROOT_DIR/.agents/workspace/data_handling_migration"

SCAN_ARGS=(--oracle-cache "$ORACLE_CACHE")
CONVERT_ARGS=(--oracle-cache "$ORACLE_CACHE" --out-store "$OUT_STORE")
VERIFY_ARGS=(--oracle-cache "$ORACLE_CACHE" --store "$OUT_STORE")

if [[ -n "$VIN_CACHE" && "$VIN_CACHE" != --* ]]; then
  SCAN_ARGS+=(--vin-cache "$VIN_CACHE")
  CONVERT_ARGS+=(--vin-cache "$VIN_CACHE")
  VERIFY_ARGS+=(--vin-cache "$VIN_CACHE")
  shift 3
else
  shift 2
fi

EXTRA_ARGS=("$@")

python "$SCRIPT_DIR/scan_legacy_offline_data.py" "${SCAN_ARGS[@]}"
python "$SCRIPT_DIR/convert_legacy_to_vin_offline.py" "${CONVERT_ARGS[@]}" "${EXTRA_ARGS[@]}"
python "$SCRIPT_DIR/verify_migrated_vin_offline.py" "${VERIFY_ARGS[@]}"

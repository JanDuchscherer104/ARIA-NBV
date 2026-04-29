#!/usr/bin/env bash
set -euo pipefail
# Sync, Download, Parse, and Materialize literature

PROJECT_ROOT=$(git rev-parse --show-toplevel)
CONFIG_PATH="${PROJECT_ROOT}/.configs/litkg.toml"

cat >&2 <<MSG
ARIA-NBV literature KG ingestion is not wired yet.

Expected config: ${CONFIG_PATH}
Expected toolkit: ${PROJECT_ROOT}/.agents/external/litkg-rs

Wire the litkg-rs CLI commands before using this target so the make task cannot
silently report a successful no-op.
MSG
exit 2

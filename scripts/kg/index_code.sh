#!/usr/bin/env bash
set -euo pipefail
# Ingest code into Neo4j using LitKG-RS CodeGraphContext
# Assumes litkg-rs is available in the environment (e.g., via cargo run)

PROJECT_ROOT=$(git rev-parse --show-toplevel)
CONFIG_PATH="${PROJECT_ROOT}/.configs/litkg.toml"

cat >&2 <<MSG
ARIA-NBV code KG indexing is not wired yet.

Expected config: ${CONFIG_PATH}
Expected toolkit: ${PROJECT_ROOT}/.agents/external/litkg-rs

Wire the litkg-rs CLI command before using this target so the make task cannot
silently report a successful no-op.
MSG
exit 2

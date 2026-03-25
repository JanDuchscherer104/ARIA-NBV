#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SKILL_DIR="${ROOT_DIR}/.agents/skills/aria-nbv-context/scripts"

exec "${SKILL_DIR}/nbv_get_context.sh" "$@"

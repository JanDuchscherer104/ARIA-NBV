#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- .agents/skills/typst-authoring docs/typst/thesis
fi

echo "== Typst authoring hygiene checks =="
echo "Targets: $*"

if command -v rg >/dev/null 2>&1; then
  echo
  echo "-- suspicious op(...) attachment followed immediately by an argument list --"
  rg -n 'op\("[^"]+"\)_[^ ]+\(' "$@" || true

  echo
  echo "-- accidental double bold or stale .codex path --"
  rg -n 'bold\(bold|\.codex/skills/typst-authoring' "$@" || true

  echo
  echo "-- temporary citation markers --"
  rg -n '\[CITATION NEEDED|TODO citation|citation needed' "$@" || true
else
  echo "ripgrep (rg) not available; skipping pattern checks" >&2
fi

echo
echo "Done. Treat matches as review prompts, not automatic failures."

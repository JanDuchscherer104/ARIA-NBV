#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

strip_ansi() {
  perl -pe 's/\e\[[0-9;]*[[:alpha:]]//g'
}

output="$(make context 2>&1 | strip_ansi)"

cat <<'EOF'
Aria-NBV startup context refresh completed.

Refreshed files under `docs/_generated/context/`:

- `source_index.md`: source-family routing map, hot-path files, local AGENTS.md
  guide inventory, preferred reveal commands, and search recipes.
- `literature_index.md`: compact index of checked-in literature source
  families, TeX/BibTeX inputs, and literature-search entrypoints.
- `data_contracts.md`: generated AST summary of Aria-NBV data/config contract
  surfaces, typed containers, factories, and key package-boundary objects.
EOF

if grep -q "^Wrote:" <<<"$output"; then
  exit 0
fi

printf 'Context refresh produced unexpected output:\n%s\n' "$output" >&2
exit 1

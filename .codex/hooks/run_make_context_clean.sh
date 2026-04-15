#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

strip_ansi() {
  perl -pe 's/\e\[[0-9;]*[[:alpha:]]//g'
}

output="$(make context 2>&1 | strip_ansi)"

cat <<'EOF'
Aria-NBV startup context refresh completed.

The hook ran `make context` and refreshed the lightweight generated context
bundle under `docs/_generated/context/`:

- `source_index.md`: source-family routing map, hot-path files, local AGENTS.md
  guide inventory, preferred reveal commands, and search recipes.
- `literature_index.md`: compact index of checked-in literature source
  families, TeX/BibTeX inputs, and literature-search entrypoints.
- `data_contracts.md`: generated AST summary of Aria-NBV data/config contract
  surfaces, typed containers, factories, and key package-boundary objects.

Use these files to localize tasks before opening broad docs, source trees, or
heavy generated context. Use `make context-heavy` only when lightweight routing
is insufficient.
EOF

printf '\nRaw refresh output:\n%s\n' "$output"

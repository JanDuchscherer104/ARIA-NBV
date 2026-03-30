#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-simplify}"
ROOT="/home/jandu/repos/NBV"
PKG_ROOT="$ROOT/aria_nbv"
SPEC="$ROOT/autoimprove.md"

cd "$PKG_ROOT"

echo "== prompt ($MODE) =="
.venv/bin/python -m aria_nbv.utils.autoimprove --spec "$SPEC" prompt --mode "$MODE"

echo
echo "== report ($MODE) =="
.venv/bin/python -m aria_nbv.utils.autoimprove --spec "$SPEC" report --mode "$MODE"

echo
echo "== score ($MODE) =="
.venv/bin/python -m aria_nbv.utils.autoimprove --spec "$SPEC" score --mode "$MODE"

#!/usr/bin/env bash
set -euo pipefail

ARIA_DSS="${1:-${ARIA_DSS:-}}"
if [[ -z "$ARIA_DSS" ]]; then
  echo "usage: $0 /dss/.../aria-nbv" >&2
  exit 2
fi

case "$ARIA_DSS" in
  "$HOME"|"$HOME"/*)
    echo "ARIA_DSS must not be inside HOME; use DSS storage for large ARIA artifacts" >&2
    exit 2
    ;;
esac

mkdir -p "$ARIA_DSS"/{data/raw,data/processed,caches/oracle,caches/vin,checkpoints,logs/slurm,logs/wandb,containers,tmp}
mkdir -p "$ARIA_DSS"/.cache/{uv,pip,huggingface,torch}

readme="$ARIA_DSS/README_ARIA_NBV_STORAGE.txt"
if [[ ! -e "$readme" ]]; then
  {
    echo "ARIA-NBV LRZ DSS layout."
    echo "Large datasets, caches, checkpoints, W&B logs, Slurm logs, containers, and temp files belong here, not in HOME."
    echo "Created: $(date -Is)"
  } > "$readme"
fi

echo "Initialized ARIA_DSS=$ARIA_DSS"

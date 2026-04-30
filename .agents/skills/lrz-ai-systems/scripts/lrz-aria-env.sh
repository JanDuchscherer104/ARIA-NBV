#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script, do not execute it: source $0 /path/to/ARIA_DSS" >&2
  exit 2
fi

export ARIA_DSS="${1:-${ARIA_DSS:-}}"
if [[ -z "$ARIA_DSS" ]]; then
  echo "Set ARIA_DSS or pass it as first argument" >&2
  return 2
fi

case "$ARIA_DSS" in
  "$HOME"|"$HOME"/*)
    echo "ARIA_DSS must not be inside HOME; use DSS storage for large ARIA artifacts" >&2
    return 2
    ;;
esac

mkdir -p "$ARIA_DSS"/{data/raw,data/processed,caches/oracle,caches/vin,checkpoints,logs/slurm,logs/wandb,containers,tmp}
mkdir -p "$ARIA_DSS"/.cache/{uv,pip,huggingface,torch}

export UV_CACHE_DIR="$ARIA_DSS/.cache/uv"
export PIP_CACHE_DIR="$ARIA_DSS/.cache/pip"
export HF_HOME="$ARIA_DSS/.cache/huggingface"
export TORCH_HOME="$ARIA_DSS/.cache/torch"
export WANDB_DIR="$ARIA_DSS/logs/wandb"
export TMPDIR="$ARIA_DSS/tmp"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "ARIA_DSS=$ARIA_DSS"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "WANDB_DIR=$WANDB_DIR"

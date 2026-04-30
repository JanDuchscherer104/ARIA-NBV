#!/usr/bin/env bash
set -euo pipefail

IMAGE="${LRZ_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
ARIA_DSS="${ARIA_DSS:-}"

if [[ -z "$ARIA_DSS" ]]; then
  echo "Set ARIA_DSS=/dss/.../aria-nbv" >&2
  exit 2
fi
if [[ ! -d "$ARIA_REPO" ]]; then
  echo "ARIA_REPO does not exist: $ARIA_REPO" >&2
  exit 2
fi
case "$ARIA_DSS" in
  "$HOME"|"$HOME"/*)
    echo "ARIA_DSS must not be inside HOME; use DSS storage for large ARIA artifacts" >&2
    exit 2
    ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside a Slurm allocation, e.g. salloc -p lrz-v100x2 --gres=gpu:1 --time=00:10:00" >&2
  exit 2
fi

mkdir -p "$ARIA_DSS" "$ARIA_DSS/tmp"

echo "Launching interactive container shell"
echo "  image: $IMAGE"
echo "  repo: $ARIA_REPO"
echo "  ARIA_DSS: $ARIA_DSS"
echo "  command: srun --pty --container-image=<image> --container-mounts=<HOME,ARIA_DSS,ARIA_REPO> bash"

exec srun --pty \
  --container-image="$IMAGE" \
  --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \
  bash -lc "cd '$ARIA_REPO' && source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh '$ARIA_DSS' && exec bash"

#!/bin/bash
#SBATCH --job-name=aria-nbv-gpu
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail

export ARIA_DSS=/ABS/PATH/TO/ARIA_DSS
export ARIA_REPO="$HOME/src/ARIA-NBV"
export LRZ_CONTAINER_IMAGE='nvcr.io#nvidia/pytorch:24.10-py3'

srun --ntasks=1 \
  --container-image="$LRZ_CONTAINER_IMAGE" \
  --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \
  bash -lc '
    set -euo pipefail
    cd "$ARIA_REPO"
    source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "$ARIA_DSS"
    nvidia-smi
    cd aria_nbv
    python -m pip install -U uv
    uv sync --all-extras
    # Inspect current console scripts before replacing this placeholder.
    uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  '

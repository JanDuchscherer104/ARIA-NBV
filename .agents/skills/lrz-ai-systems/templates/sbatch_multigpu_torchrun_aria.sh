#!/bin/bash
#SBATCH --job-name=aria-nbv-ddp
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail

export ARIA_DSS=/ABS/PATH/TO/ARIA_DSS
export ARIA_REPO="$HOME/src/ARIA-NBV"
export LRZ_CONTAINER_IMAGE='nvcr.io#nvidia/pytorch:24.10-py3'

srun --ntasks=1 --ntasks-per-node=1 \
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
    torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE}" <TRAIN_MODULE_OR_SCRIPT> <ARGS>
  '

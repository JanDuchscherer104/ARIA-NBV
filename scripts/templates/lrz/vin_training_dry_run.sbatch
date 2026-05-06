#!/usr/bin/env bash
#SBATCH --job-name=aria-vin-train-dryrun
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail

ARIA_DSS="${ARIA_DSS:-/ABS/PATH/TO/ARIA_DSS}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
LRZ_CONTAINER_IMAGE="${LRZ_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"
RUN_ID="${RUN_ID:-vin-training-dryrun-YYYYMMDD}"
DATASET_VERSION="${DATASET_VERSION:-VIN_OFFLINE_STORE_VERSION_PLACEHOLDER}"
JOB_ID="${SLURM_JOB_ID:-manual}"
GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-2}"
INPUT_STORE="$ARIA_DSS/caches/vin/$DATASET_VERSION"
CHECKPOINT_ROOT="$ARIA_DSS/checkpoints/vin/$RUN_ID"
WANDB_ROOT="$ARIA_DSS/logs/wandb"
TEMP_ROOT="$ARIA_DSS/tmp/$JOB_ID/vin-training/$RUN_ID"

cat <<DRYRUN
ARIA-NBV LRZ VIN training dry run
=================================

This template does not train a VIN model. It documents the Slurm/Pyxis
single-node multi-GPU launch shape and expected DSS paths.

Slurm:
  job_id: $JOB_ID
  partition_hint: single GPU smoke first, then one-node multi-GPU after input checks
  gpu_request: --gres=gpu:2
  gpus_on_node: $GPUS_ON_NODE

Enroot/Pyxis:
  image: $LRZ_CONTAINER_IMAGE
  container_mounts: $HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO

DSS:
  input_store: $INPUT_STORE
  checkpoint_root: $CHECKPOINT_ROOT
  wandb_root: $WANDB_ROOT
  temp_root: $TEMP_ROOT
  completion_marker: $CHECKPOINT_ROOT/_TRAINING_DRY_RUN.json

Dry-run Pyxis launch shape:
  srun --ntasks=1 --ntasks-per-node=1 \\
    --container-image="$LRZ_CONTAINER_IMAGE" \\
    --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \\
    bash -lc 'cd "$ARIA_REPO/aria_nbv" && torchrun --standalone \\
      --nproc_per_node="$GPUS_ON_NODE" \\
      <VIN_TRAINING_MODULE_OR_SCRIPT> \\
      --offline-store "$INPUT_STORE" \\
      --checkpoint-root "$CHECKPOINT_ROOT" \\
      --wandb-dir "$WANDB_ROOT" \\
      --dry-run'

Resume and atomic-write expectations:
  - checkpoints belong under the run-specific checkpoint root
  - write checkpoint temp files before publishing latest/best pointers
  - do not mutate immutable VIN offline stores from the training job
  - restart from explicit checkpoint paths rather than scanning broad DSS roots
DRYRUN

#!/usr/bin/env bash
#SBATCH --job-name=aria-oracle-dryrun
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --array=0-0
#SBATCH --output=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%A_%a.out
#SBATCH --error=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%A_%a.err

set -euo pipefail

ARIA_DSS="${ARIA_DSS:-/ABS/PATH/TO/ARIA_DSS}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
LRZ_CONTAINER_IMAGE="${LRZ_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"
RUN_ID="${RUN_ID:-oracle-dryrun-YYYYMMDD}"
DATASET_VERSION="${DATASET_VERSION:-ASE_VERSION_PLACEHOLDER}"
SHARD_MANIFEST="${SHARD_MANIFEST:-$ARIA_DSS/data/staging/manifests/oracle_shards.jsonl}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"
FINAL_SHARD_DIR="$ARIA_DSS/caches/oracle/staging/$RUN_ID/shards/shard-$SHARD_ID"
TEMP_SHARD_DIR="$ARIA_DSS/tmp/$JOB_ID/oracle/$RUN_ID/shard-$SHARD_ID.tmp"

cat <<DRYRUN
ARIA-NBV LRZ oracle generation dry run
======================================

This template does not generate oracle labels. It documents the Slurm/Pyxis
shape and expected DSS paths for a future implementation.

Slurm:
  job_id: $JOB_ID
  array_task_id: $SHARD_ID
  partition_hint: lrz-v100x2 for smoke, A100/H100 80GB class partitions for scale
  gpu_request: --gres=gpu:1

Enroot/Pyxis:
  image: $LRZ_CONTAINER_IMAGE
  container_mounts: $HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO

DSS:
  dataset_version: $DATASET_VERSION
  shard_manifest: $SHARD_MANIFEST
  temp_shard_dir: $TEMP_SHARD_DIR
  final_shard_dir: $FINAL_SHARD_DIR
  completion_marker: $FINAL_SHARD_DIR/_SUCCESS.json
  owner_sidecar: $FINAL_SHARD_DIR/_owner.json

Dry-run Pyxis launch shape:
  srun --ntasks=1 \\
    --container-image="$LRZ_CONTAINER_IMAGE" \\
    --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \\
    bash -lc 'cd "$ARIA_REPO/aria_nbv" && uv run <ORACLE_GENERATION_ENTRYPOINT> \\
      --dataset-version "$DATASET_VERSION" \\
      --shard-manifest "$SHARD_MANIFEST" \\
      --shard-id "$SHARD_ID" \\
      --output-tmp "$TEMP_SHARD_DIR" \\
      --output-final "$FINAL_SHARD_DIR" \\
      --dry-run'

Resume and atomic-write expectations:
  - one array task owns this final shard path
  - skip only when the completion marker exists and validates
  - write heavy output to the temp path first
  - promote temp to final by same-filesystem rename
  - write the completion marker last
DRYRUN

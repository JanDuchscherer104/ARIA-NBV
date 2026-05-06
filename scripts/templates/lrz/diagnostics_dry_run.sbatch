#!/usr/bin/env bash
#SBATCH --job-name=aria-diag-dryrun
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
RUN_ID="${RUN_ID:-diagnostics-dryrun-YYYYMMDD}"
DATASET_VERSION="${DATASET_VERSION:-ARTIFACT_VERSION_PLACEHOLDER}"
SHARD_MANIFEST="${SHARD_MANIFEST:-$ARIA_DSS/data/staging/manifests/diagnostics_shards.jsonl}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"
REPORT_DIR="$ARIA_DSS/logs/diagnostics/$RUN_ID/shard-$SHARD_ID"
TEMP_REPORT_DIR="$ARIA_DSS/tmp/$JOB_ID/diagnostics/$RUN_ID/shard-$SHARD_ID.tmp"

cat <<DRYRUN
ARIA-NBV LRZ diagnostics dry run
================================

This template does not render diagnostics, write Rerun recordings, or validate
generated stores. It documents the Slurm/Pyxis shape and expected report paths.

Slurm:
  job_id: $JOB_ID
  array_task_id: $SHARD_ID
  partition_hint: lrz-cpu for manifest-only checks, lrz-v100x2 when GPU rendering is needed
  gpu_request: --gres=gpu:1

Enroot/Pyxis:
  image: $LRZ_CONTAINER_IMAGE
  container_mounts: $HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO

DSS:
  artifact_version: $DATASET_VERSION
  shard_manifest: $SHARD_MANIFEST
  temp_report_dir: $TEMP_REPORT_DIR
  final_report_dir: $REPORT_DIR
  completion_marker: $REPORT_DIR/_SUCCESS.json
  owner_sidecar: $REPORT_DIR/_owner.json

Dry-run Pyxis launch shape:
  srun --ntasks=1 \\
    --container-image="$LRZ_CONTAINER_IMAGE" \\
    --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \\
    bash -lc 'cd "$ARIA_REPO/aria_nbv" && uv run <DIAGNOSTICS_ENTRYPOINT> \\
      --artifact-version "$DATASET_VERSION" \\
      --shard-manifest "$SHARD_MANIFEST" \\
      --shard-id "$SHARD_ID" \\
      --report-tmp "$TEMP_REPORT_DIR" \\
      --report-final "$REPORT_DIR" \\
      --dry-run'

Resume and atomic-write expectations:
  - diagnostics should read generated artifacts without mutating them
  - write reports to temp first, validate, rename, then write completion marker
  - record failed shard diagnostics separately from completed report roots
DRYRUN

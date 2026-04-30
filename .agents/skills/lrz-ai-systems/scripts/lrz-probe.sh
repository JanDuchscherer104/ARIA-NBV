#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname || true

echo "== user =="
id || true
groups || true

echo "== pwd/home =="
pwd || true
echo "HOME=${HOME:-<unset>}"

echo "== slurm allocation =="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-<none>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<none>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<none>}"

echo "== DSS containers =="
if command -v dssusrinfo >/dev/null 2>&1; then
  dssusrinfo all || true
else
  echo "dssusrinfo not found"
fi

echo "== partitions summary =="
echo "warning: run Slurm status commands as one-shot checks; do not poll in tight loops"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%20P %8a %12l %6D %8t %30G %N" || true
else
  echo "sinfo not found"
fi

echo "== my jobs =="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER:-}" -o "%.18i %.20P %.30j %.8T %.10M %.6D %R" || true
else
  echo "squeue not found"
fi

echo "== gpu visibility =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; expected on GPU compute nodes or inside GPU containers"
fi

#!/usr/bin/env bash
set -euo pipefail

PARTITION="${LRZ_PARTITION:-lrz-hgx-h100-94x4}"
GPUS="${LRZ_GPUS:-2}"
NODES="${LRZ_NODES:-1}"
TIME="${LRZ_TIME:-04:00:00}"
CPUS="${LRZ_CPUS:-32}"
MEM="${LRZ_MEM:-240G}"
IMAGE="${LRZ_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
ARIA_DSS="${ARIA_DSS:-}"
TRAIN_CMD="${*:-}"

if [[ -z "$ARIA_DSS" || -z "$TRAIN_CMD" ]]; then
  echo "usage: ARIA_DSS=/dss/.../aria-nbv LRZ_GPUS=2 $0 '<python module/script and args after torchrun>'" >&2
  exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LRZ_GPUS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$NODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "LRZ_NODES must be a positive integer" >&2
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

mkdir -p "$ARIA_DSS/logs/slurm"
jobfile="$(mktemp --suffix=.sbatch)"
cat > "$jobfile" <<SBATCH
#!/bin/bash
#SBATCH --job-name=aria-nbv-ddp
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=$ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail

export ARIA_DSS='$ARIA_DSS'
export ARIA_REPO='$ARIA_REPO'
export LRZ_CONTAINER_IMAGE='$IMAGE'
export LRZ_GPUS='$GPUS'

srun --ntasks=$NODES --ntasks-per-node=1 \\
  --container-image="\$LRZ_CONTAINER_IMAGE" \\
  --container-mounts="\$HOME:\$HOME,\$ARIA_DSS:\$ARIA_DSS,\$ARIA_REPO:\$ARIA_REPO" \\
  bash -lc 'set -euo pipefail; cd "$ARIA_REPO"; source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "$ARIA_DSS"; nvidia-smi; cd aria_nbv; python -m pip install -U uv; uv sync --all-extras; torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-$LRZ_GPUS}" $TRAIN_CMD'
SBATCH

echo "Prepared multi-GPU sbatch job"
echo "  partition: $PARTITION"
echo "  nodes: $NODES"
echo "  gres: gpu:$GPUS"
echo "  time: $TIME"
echo "  repo: $ARIA_REPO"
echo "  ARIA_DSS: $ARIA_DSS"
echo "  jobfile: $jobfile"
echo "  command: sbatch $jobfile"
sbatch "$jobfile"

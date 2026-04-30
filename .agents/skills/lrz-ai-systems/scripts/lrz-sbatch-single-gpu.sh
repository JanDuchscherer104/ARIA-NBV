#!/usr/bin/env bash
set -euo pipefail

PARTITION="${LRZ_PARTITION:-lrz-v100x2}"
GPUS="${LRZ_GPUS:-1}"
TIME="${LRZ_TIME:-01:00:00}"
CPUS="${LRZ_CPUS:-16}"
MEM="${LRZ_MEM:-120G}"
IMAGE="${LRZ_CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
ARIA_DSS="${ARIA_DSS:-}"
CMD="${*:-}"

if [[ -z "$ARIA_DSS" || -z "$CMD" ]]; then
  cat >&2 <<USAGE
usage: ARIA_DSS=/dss/.../aria-nbv $0 '<command inside aria_nbv>'
optional env: LRZ_PARTITION, LRZ_GPUS, LRZ_TIME, LRZ_CPUS, LRZ_MEM, LRZ_CONTAINER_IMAGE, ARIA_REPO
example: ARIA_DSS=/dss/.../aria-nbv LRZ_PARTITION=lrz-v100x2 LRZ_TIME=00:30:00 $0 'uv run python -c "import torch; print(torch.cuda.is_available())"'
USAGE
  exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LRZ_GPUS must be a positive integer" >&2
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
#SBATCH --job-name=aria-nbv-gpu
#SBATCH --partition=$PARTITION
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

echo "Start \$(date -Is) on \$(hostname) job=\${SLURM_JOB_ID:-unknown}"

srun --ntasks=1 \\
  --container-image="\$LRZ_CONTAINER_IMAGE" \\
  --container-mounts="\$HOME:\$HOME,\$ARIA_DSS:\$ARIA_DSS,\$ARIA_REPO:\$ARIA_REPO" \\
  bash -lc 'set -euo pipefail; cd "$ARIA_REPO"; source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "$ARIA_DSS"; nvidia-smi; cd aria_nbv; python -m pip install -U uv; uv sync --all-extras; $CMD'
SBATCH

echo "Prepared single-GPU sbatch job"
echo "  partition: $PARTITION"
echo "  gres: gpu:$GPUS"
echo "  time: $TIME"
echo "  repo: $ARIA_REPO"
echo "  ARIA_DSS: $ARIA_DSS"
echo "  jobfile: $jobfile"
echo "  command: sbatch $jobfile"
sbatch "$jobfile"

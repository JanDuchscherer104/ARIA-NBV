#!/usr/bin/env bash
set -euo pipefail

TIME="${LRZ_TIME:-02:00:00}"
CPUS="${LRZ_CPUS:-16}"
MEM="${LRZ_MEM:-120G}"
ARIA_REPO="${ARIA_REPO:-$HOME/src/ARIA-NBV}"
ARIA_DSS="${ARIA_DSS:-}"
CMD="${*:-}"

if [[ -z "$ARIA_DSS" || -z "$CMD" ]]; then
  echo "usage: ARIA_DSS=/dss/.../aria-nbv $0 '<cpu command inside aria_nbv>'" >&2
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
#SBATCH --job-name=aria-nbv-cpu
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=$ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail
export ARIA_DSS='$ARIA_DSS'
export ARIA_REPO='$ARIA_REPO'

cd "\$ARIA_REPO"
source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "\$ARIA_DSS"
cd aria_nbv
python -m pip install -U uv
uv sync --all-extras
$CMD
SBATCH

echo "Prepared CPU sbatch job"
echo "  partition: lrz-cpu"
echo "  qos: cpu"
echo "  time: $TIME"
echo "  repo: $ARIA_REPO"
echo "  ARIA_DSS: $ARIA_DSS"
echo "  jobfile: $jobfile"
echo "  command: sbatch $jobfile"
sbatch "$jobfile"

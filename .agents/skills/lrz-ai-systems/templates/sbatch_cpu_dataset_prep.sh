#!/bin/bash
#SBATCH --job-name=aria-nbv-cpu-prep
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.out
#SBATCH --error=/ABS/PATH/TO/ARIA_DSS/logs/slurm/%x-%j.err

set -euo pipefail

export ARIA_DSS=/ABS/PATH/TO/ARIA_DSS
export ARIA_REPO="$HOME/src/ARIA-NBV"

cd "$ARIA_REPO"
source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "$ARIA_DSS"
cd aria_nbv
python -m pip install -U uv
uv sync --all-extras

# Replace with the actual CPU data prep command after inspecting current ARIA CLI entry points.
uv run pytest tests/test_panels_dispatcher.py -q

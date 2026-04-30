# Containers With Pyxis

LRZ AI Systems use Enroot on compute nodes. Enroot is not available on SSH login nodes. Prefer Pyxis Slurm options instead of direct Enroot commands in ARIA job scripts.

## Rules

- Run containers only inside Slurm allocations or batch jobs.
- Prefer NGC PyTorch containers for GPU workloads.
- Mount `$HOME`, `$ARIA_DSS`, and the ARIA repo explicitly.
- Put package, model, temp, and W&B caches on DSS by sourcing `scripts/lrz-aria-env.sh`.
- Keep NGC credentials outside git.

## Credential Location

Place NGC credentials in `~/enroot/.credentials` on LRZ. Do not commit that file or paste tokens into repo files.

## Interactive Container Shell

```bash
salloc -p lrz-v100x2 --gres=gpu:1 --time=00:30:00
export ARIA_DSS=/dss/.../aria-nbv
export LRZ_CONTAINER_IMAGE='nvcr.io#nvidia/pytorch:24.10-py3'
.agents/skills/lrz-ai-systems/scripts/lrz-container-shell.sh
```

## Pyxis Pattern

```bash
srun --ntasks=1 \
  --container-image="$LRZ_CONTAINER_IMAGE" \
  --container-mounts="$HOME:$HOME,$ARIA_DSS:$ARIA_DSS,$ARIA_REPO:$ARIA_REPO" \
  bash -lc 'cd "$ARIA_REPO" && source .agents/skills/lrz-ai-systems/scripts/lrz-aria-env.sh "$ARIA_DSS" && exec bash'
```

---
name: lrz-ai-systems
description: "Use when working with LRZ AI Systems remote compute for ARIA-NBV: SSH/login.ai.lrz.de, DSS storage, Slurm GPU/CPU allocations, Enroot/Pyxis containers, dataset/cache/training batch jobs, or debugging remote job failures."
metadata:
  applies_to:
    - ".configs/lrz/**"
    - "scripts/templates/**"
    - ".agents/skills/lrz-ai-systems/**"
    - "docs/contents/impl/lrz_dry_runs.qmd"
  triggers:
    - "LRZ"
    - "Slurm"
    - "DSS"
    - "Pyxis"
  must_read:
    - ".agents/skills/lrz-ai-systems/references/cheatsheet.md"
    - ".agents/skills/lrz-ai-systems/references/storage-dss.md"
    - ".agents/skills/lrz-ai-systems/references/slurm-partitions.md"
  verification:
    - "shellcheck or dry-run checks for changed scripts where available"
    - "make check-agent-memory for LRZ guidance changes"
---

# LRZ AI Systems

## Read First

- `references/cheatsheet.md` for one-shot operator commands.
- `references/storage-dss.md` before creating datasets, caches, checkpoints, logs, or containers.
- `references/slurm-partitions.md` before requesting CPU/GPU resources.
- `references/containers-pyxis.md` before launching container jobs.
- `references/aria-workflows.md` before ARIA dataset/cache/training runs.
- `references/service-desk-templates.md` when access, DSS quota, or project membership is missing.

## Rules

- Use `login.ai.lrz.de` login nodes only for editing, transfer, inspection, and Slurm submission.
- Never run heavy computation on login nodes.
- Never use `sudo`.
- Keep code, small config, SSH/git config, and non-committed NGC credentials in `$HOME`.
- Keep ASE/ATEK shards, oracle RRI caches, VIN offline stores, checkpoints, Slurm logs, W&B logs, containers, temp files, and package/model caches under `$ARIA_DSS`.
- Never submit GPU work without `--gres=gpu:<N>`.
- Do not automate `sinfo`, `squeue`, `sacct`, or similar Slurm polling loops.
- Use current `sinfo` output at runtime; partition names and limits can change.
- Treat `mix` nodes as partially allocated, not necessarily full.
- Use `lrz-v100x2` for GPU smoke tests, `lrz-cpu` for CPU preprocessing, and H100/A100 80GB partitions for serious cache/training work.
- Avoid MCML partitions unless access/QOS is confirmed by project membership or a short successful test allocation.
- Use Pyxis Slurm options (`--container-image`, `--container-mounts`) for containers on compute nodes.
- Do not rely on host conda/pip as the supported LRZ workload environment.

## Standard Workflow

1. Inspect access and storage with `scripts/lrz-probe.sh` and `dssusrinfo all`.
2. Choose or request an AI Systems DSS container, then run `scripts/lrz-dss-init.sh "$ARIA_DSS"`.
3. Inspect partitions once with `scripts/lrz-resources.sh summary` or `scripts/lrz-resources.sh gpu`.
4. Smoke test GPU access with `salloc -p lrz-v100x2 --gres=gpu:1 --time=00:10:00` and `srun --pty bash`.
5. Use `scripts/lrz-container-shell.sh` inside an allocation for interactive container debugging.
6. Use `scripts/lrz-sbatch-cpu.sh`, `scripts/lrz-sbatch-single-gpu.sh`, or `scripts/lrz-sbatch-multigpu.sh` for batch work.
7. Inspect current ARIA CLI entry points before filling cache/training commands:

```bash
cd aria_nbv
uv run python - <<'PY'
import importlib.metadata as m
for ep in m.entry_points().select(group="console_scripts"):
    if "nbv" in ep.name or "aria" in ep.name:
        print(ep.name, "->", ep.value)
PY
```

## Verification

- `bash -n .agents/skills/lrz-ai-systems/scripts/*.sh`
- `find .agents/skills/lrz-ai-systems/scripts -maxdepth 1 -type f -perm -111`
- Run a secret/path scan for credential terms, personal LRZ usernames, fixed DSS paths, and host-local absolute paths.
- `python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/lrz-ai-systems`
- `make check-agent-memory`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

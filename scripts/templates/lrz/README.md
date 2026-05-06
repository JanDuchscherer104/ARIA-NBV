# LRZ Dry-Run Templates

These Slurm templates document the expected LRZ launch shape for future ARIA-NBV
scale jobs. They are deliberately dry-run only: after replacing the SBATCH log
path placeholders, submitting one should print the paths, ownership contract,
and placeholder command, then exit without generating oracle labels, rollouts,
VIN checkpoints, or diagnostics artifacts.

Before converting any template into a real job:

1. Inspect current ARIA console entry points from the LRZ checkout.
2. Replace placeholder commands such as `<ORACLE_GENERATION_ENTRYPOINT>`.
3. Confirm `ARIA_DSS`, quota, inode pressure, and the shard manifest.
4. Keep all large outputs, logs, caches, checkpoints, and containers under
   `$ARIA_DSS`.
5. Preserve the atomic-write and resume expectations documented in
   `.configs/lrz/README.md`.

## Template Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARIA_DSS` | `/ABS/PATH/TO/ARIA_DSS` | DSS root for large ARIA artifacts. |
| `ARIA_REPO` | `$HOME/src/ARIA-NBV` | LRZ checkout path. |
| `LRZ_CONTAINER_IMAGE` | `nvcr.io#nvidia/pytorch:24.10-py3` | Enroot/Pyxis container image URI. |
| `RUN_ID` | workflow-specific dry-run ID | Run namespace for logs and staging. |
| `DATASET_VERSION` | placeholder | Dataset/cache/offline-store version. |
| `SHARD_MANIFEST` | workflow-specific path | Deterministic shard manifest. |

The templates use Slurm comments and Pyxis options as documentation. They do not
call `srun` by default, because the current task is to define the dry-run
contract rather than to launch containers or implement generation code.

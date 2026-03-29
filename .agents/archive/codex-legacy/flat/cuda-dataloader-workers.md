# Task: Diagnose CUDA error after increasing DataLoader workers

## Findings
- The DataLoader workers are executing GPU code inside `VinOracleIterableDataset.__iter__`, via `OracleRriLabeler` → `CandidateViewGenerator.generate_from_typed_sample` which calls `.to(device="cuda")` on occupancy/poses. This happens inside the worker subprocess.
- `Pytorch3DDepthRenderer` also defaults to CUDA and is invoked during labeler execution, so workers attempt to initialize CUDA.
- `AriaNBVExperiment.setup_target` calls `torch.cuda.manual_seed_all(...)` before spawning DataLoader workers, so CUDA is already initialized in the parent. Forked workers then fail with `RuntimeError: Cannot re-initialize CUDA in forked subprocess.`
- `VinDataModuleConfig.num_workers` is documented to “keep at 0; labeler is not multiprocessing-friendly,” which aligns with the observed failure.

## Suggestions
- Keep `num_workers=0` for the online oracle labeler path (run GPU label generation in the main process).
- If parallelism is required, consider using `multiprocessing_context="spawn"` for the DataLoader and avoid CUDA initialization in the parent process before workers start (or move all CUDA ops out of workers), but expect high GPU contention.
- Alternatively, force the labeler/renderer device to CPU for DataLoader workers if GPU usage is not required there.

## Notes
- No code changes were made; this is a diagnosis-only task.

## Changes Applied
- Added `multiprocessing_context` (default "spawn") to `VinDataModuleConfig` and wired it into DataLoader creation.
- Added `_multiprocessing_context()` helper to guard context usage when `num_workers=0`.
- Skipped `torch.cuda.manual_seed_all` when `num_workers>0` to avoid CUDA init before worker start.

## Follow-up Diagnosis (2025-12-24)
- `oracle_rri/oracle_rri/lightning/lit_datamodule.py` currently shows `num_workers=0` and no `multiprocessing_context` field, so the earlier spawn-based changes appear to have been reverted or not present in this working tree.
- The error persists because the DataLoader still uses the default `fork` start method, and the labeler stack (`CandidateViewGenerator` and `Pytorch3DDepthRenderer`) tries to use CUDA in workers.
- To make multiproc safe, either enforce `spawn` globally very early (CLI entrypoint) + pass `multiprocessing_context="spawn"` into DataLoader, or avoid CUDA in workers (force CPU in labeler when `get_worker_info()` is set).

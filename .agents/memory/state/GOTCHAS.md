---
id: gotchas
updated: 2026-03-30
scope: repo
owner: jan
status: active
tags: [workflow, training, cache, frames]
---

# Gotchas

## Environment and Tooling
- Prefer `uv run --project aria_nbv pytest` or `aria_nbv/.venv/bin/python -m pytest`; the system interpreter may miss dependencies such as `power_spherical`.
- Assume the environment is working unless the user indicates otherwise, but verify the exact interpreter before concluding a dependency problem.
- Oracle RRI defaults to the `pytorch3d_cuda` profile even on Apple hosts. A broken local PyTorch3D binary, such as an ABI error mentioning `TensorImpl::decref_pyobject`, is a local setup failure for that selected profile, not permission to silently switch core defaults to Mojo.
- Streamlit runs page code in a worker thread; the Python-imported Mojo collision kernels can segfault there. The Apple profile keeps direct in-process Mojo kernels for CLI/tests, but Streamlit routes the collision, render, point-cloud, and oracle-distance Mojo stages through subprocesses in that worker-thread context.
- Streamlit worker threads are also unsafe for the Python-imported Mojo render and oracle-distance kernels. The app now runs Mojo collision, depth, point-cloud, and oracle-distance stages in subprocesses from that context, while direct CLI/tests keep the in-process Mojo path.
- `make context` refreshes the lightweight routing artifacts only; use targeted search on `source_index.md`, `literature_index.md`, and `data_contracts.md` instead of loading broad dumps.
- `make context-heavy` and the `context-uml`, `context-docstrings`, or `context-tree` targets are explicit fallback tools for architecture or refactor tasks.

## Training and Validation
- Validation is disabled by default unless `trainer_config.enable_validation=true`; otherwise Lightning forces `limit_val_batches=0` and `check_val_every_n_epoch=0`.
- Treat explicit user termination criteria as binding. If they imply stronger verification, expand the test plan rather than stopping at a smoke test.
- Prefer real-data or integration-style verification when feasible for package changes; do not rely only on mocks for end-to-end behavior claims.

## Offline Cache and Splits
- Offline cache splits are file-backed; `train_index.jsonl` and `val_index.jsonl` may be created or updated when loading caches with `split="train"` or `split="val"`.
- Rebuild indices from `samples/*.pt` with `rebuild_cache_index(cache_dir=..., train_val_split=..., rng_seed=...)` in `oracle_rri.data.offline_cache` when split state becomes stale.
- Training/validation from cache uses `OracleRriCacheDatasetConfig.train_val_split` together with `split="train"` / `split="val"`; `VinDataModule` auto-derives `val_cache` when split mode is active.
- `VinOracleBatch.collate` expects cache-ready `VinSnippetView` instances rather than raw `EfmSnippetView` samples.

## Frames and Geometry
- Pose-frame consistency and CW90 corrections are easy to misuse across rendering and VIN inputs.
- Use `PoseTW` and `CameraTW` instead of raw matrices in normal package code.
- Document tensor shapes and coordinate frames when a contract is not obvious from the type alone.

## EVL / OBB
- EVL OBB outputs are not batch-collatable yet; entity-aware runs may need `batch_size=None` or OBB outputs disabled.
- Candidate validity heuristics and semidense visibility proxies are conservative; do not assume they are equivalent to training masks unless the training loop explicitly applies them.

## Config and Pydantic
- `Field(default=<callable>)` stores the callable itself; use `Field(default_factory=...)` for computed defaults.
- Prefer config-local `field_validator` and `model_validator` hooks for cross-field validation and coercion.

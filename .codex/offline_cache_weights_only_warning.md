Context: Silence PyTorch torch.load FutureWarning when reading offline cache payloads.

Change:
- Update `oracle_rri/oracle_rri/data/offline_cache.py` to call
  `torch.load(..., weights_only=False)` inside a `warnings.catch_warnings()` block
  that suppresses the FutureWarning.

Why:
- `weights_only=True` fails on cached payloads containing NumPy scalars, so we
  retain full pickle loading for trusted caches while silencing the noisy warning.

Tests:
- `ruff format oracle_rri/oracle_rri/data/offline_cache.py` (no changes).
- `ruff check oracle_rri/oracle_rri/data/offline_cache.py` failed due to pre-existing lint issues in that file (not introduced here).
- `pytest tests/data/test_offline_cache.py` skipped (missing real data/EVL assets in this environment).

Notes:
- To fully validate with real data, run `pytest tests/data/test_offline_cache.py` in an environment with `.data/ase_efm`, `.data/ase_meshes`, and EVL checkpoints present.

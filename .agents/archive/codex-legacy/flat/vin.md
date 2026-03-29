# VIN (View Introspection Network) — status + notes

## What was fixed/restored

- Restored `docs/contents/impl/vin_nbv.qmd` (it had been truncated to 0 bytes) and verified it renders via Quarto.
- Extended `docs/typst/slides/slides_3.typ` with a VIN section (I/O, EVL features, encodings, binning, CORAL, open questions) and a compiled-in shape summary slide.
- Added `oracle_rri/scripts/summarize_vin.py` to generate a PyTorch forward-hook shape/parameter summary on a real snippet (used for the slides).
- Ensured VIN uses the review-aligned pose descriptor + encodings:
  - shell descriptor from candidate `T_camera_rig` (camera←reference rig),
  - **spherical harmonics** for the two unit directions (`u` position direction, `f` forward direction) via **e3nn**,
  - **1D Fourier features for radius** (default: encode `log(r+eps)`).
- Removed any dependence on `so3log`/`so3log_map` from the default VIN path.
- Ensured CORAL comes from the upstream reference implementation (`coral-pytorch`).
- Added a hard guard in `OracleRriLabeler` for the “0 candidates” failure mode (fails fast with a helpful message).

## Code map (current)

- `oracle_rri/oracle_rri/vin/backbone_evl.py`
  - Single adapter that touches EVL instantiation + forward and returns a minimal feature contract.
- `oracle_rri/oracle_rri/vin/model.py`
  - Pure scorer head: pose encoding + voxel queries + MLP + CORAL; supports passing `candidate_poses_camera_rig`.
- `oracle_rri/oracle_rri/vin/spherical_encoding.py`
  - `ShellShPoseEncoder`: SH(u,f) via e3nn + 1D Fourier radius embedding + scalar MLP.
- `oracle_rri/oracle_rri/vin/coral.py`
  - Wraps `coral-pytorch` (`CoralLayer`, `coral_loss`, `expected` utilities).
- `oracle_rri/scripts/train_vin.py`
  - Minimal end-to-end training smoke test that uses `OracleRriLabeler` online.

## Validation performed (real data)

- Smoke training run succeeded end-to-end on local ASE shards + meshes:
  - binner fit → labeler → EVL forward (frozen) → VIN head forward → CORAL loss → optimizer step → checkpoint saved.
- Pytest integration tests (real data) passed:
  - `oracle_rri/tests/integration/test_vin_real_data.py`
  - `oracle_rri/tests/integration/test_oracle_rri_labeler_real_data.py`

## Dependency notes

- `oracle_rri/pyproject.toml` now includes:
  - `e3nn==0.5.1`
  - `coral-pytorch==1.4.0`
- `oracle_rri/uv.lock` updated via `uv sync`.

## Remaining TODOs / recommended next patches

- Candidate-conditioned voxel query:
  - Replace “sample at camera center” with **frustum point sampling** (K points/rays/depths) and pool mean+max.
  - Add strict point-level validity masking (invalid samples must not contribute to pooling).
- Training metrics:
  - Log Spearman rank correlation and top-k recall between predicted score and oracle RRI.
- Threshold versioning:
  - Version binner edges by (candidate-generator config hash, EVL checkpoint id, num_bins).
- Data efficiency:
  - Online oracle labels are expensive; add caching (depths/PCs or oracle RRIs) for real training runs.

## Known pitfalls

- `CandidateDepthRenderer` can return candidates with 0 valid pixels when the oversampled batch is already ≤ `max_candidates_final`.
  - Training should consider masking by per-candidate valid-pixel counts (in addition to `candidate_valid` + finite RRI).

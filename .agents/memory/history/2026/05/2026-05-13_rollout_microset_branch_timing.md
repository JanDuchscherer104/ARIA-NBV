---
id: 2026-05-13_rollout_microset_branch_timing
date: 2026-05-13
title: "Rollout Microset Branch Timing"
status: done
topics: [rollouts, target-rri, rerun, runtime]
confidence: high
canonical_updates_needed: []
artifacts:
  - .data/offline_cache/rollouts_v1_microset.zarr
  - .artifacts/rerun/rollout_v1_microset_000.rrd
---

## Task

Generate a small V1 rollout evidence microset, make branch counts stochastic
while preserving deterministic branch schedules, and inspect runtime bottlenecks.

## Findings

The local workstation has CUDA-visible PyTorch, but the installed PyTorch3D
extensions are CPU-only for rasterization and point-mesh distance. Rollout
configs therefore use CPU for the trusted local smoke path until a CUDA-enabled
PyTorch3D build is available.

The dominant cost is repeated target scorer evaluation over retained rollout
frontier nodes. The branch model is not simply "sample K once." For every
frontier node, the generator regenerates a full candidate table, scores every
valid candidate, samples or greedily selects branch_count actions, and keeps the
top beam_width chains. With horizon 2 and root branch_count 2, this means one
root scoring call plus one scoring call for each retained child. For the
microset with 32 candidate shells, horizon 2, beam width 2, and seeded
stochastic branch choices, the generated target produced 3 expanded nodes, 21
valid scored candidates, 31.723 s in candidate generation, 37.202 s in target
scoring, and 0.003 s in selection. The root target scorer call spent 16.943 s
in depth rendering and 10.156 s in target oracle scoring.

The first stochastic branch schedule sampled branch counts [2, 1, 1] under
seed 0 and probabilities [0.75, 0.25] over branch factors [1, 2]. This produced
two root children and one child expansion each. That is valid for diversity,
but it also shows why stochastic branch factors should be reported in store
lineage or run logs whenever they are used for evidence, because the actual
expanded-node count is part of the compute budget.

The generated store validates with 1 source, 1 matched target, 2 rollout chains,
4 steps, 128 candidate rows, 36 actor-valid/trainable candidates, 4 selected
candidates, and 0 selected-invalid actions. Target RRI is finite on every
trainable candidate. Scene RRI was intentionally disabled in the microset after
a one-row full-scene audit run exceeded several minutes on the CPU-only
PyTorch3D path.

The Rerun recording was regenerated with the current SDK and contains GT mesh,
detected/GT OBB layers, target-highlight metadata, rollout step cameras, valid
and invalid candidate layers, selected path, selected target-RRI scalar traces,
and rollout metadata. Screenshot export could not be verified on this machine
because it is headless and lacks `DISPLAY`/`WAYLAND_DISPLAY`; the `.rrd` itself
prints successfully with the current `rerun rrd print` command.

## Interpretation

The microset is evidence-grade for schema and protocol debugging, not for
training-scale claims. It validates the V1 contracts that matter before Q_H:
actor-visible target selection, GT-only target label/evaluation fields, hard
candidate masks, finite target RRI on trainable candidates, selected-transition
derivability through `q_h_view()`, and Rerun-inspectable geometry.

The runtime bottleneck points to two separate problems:

- Candidate generation costs about 10 s per expanded node on this sample, even
  before scoring. That likely includes repeated full-shell sampling plus
  collision/free-space checks against the mesh.
- Target scoring is still dominated by CPU PyTorch3D rendering and point-mesh
  distance. Full-scene RRI is much more expensive than target-cropped RRI on the
  same hardware and should stay an optional audit until it has a downsampled or
  GPU-backed path.

## Follow-Up Recommendations

- Install or build CUDA-enabled PyTorch3D before scaling rollout generation on
  this workstation; otherwise move full-scale generation to LRZ with a verified
  CUDA PyTorch3D container.
- Treat expanded node count as an explicit budget metric next to horizon,
  candidate count, branch factor, and beam width. A deterministic
  `branch_factor_schedule = [1, 1, ...]` is the cheapest trusted smoke path;
  stochastic branch counts are useful for diversity but must be logged.
- Add a scene-RRI audit mode that downsamples the scene points/mesh or evaluates
  only selected candidates. Do not enable full-scene RRI by default in local
  microset configs until that path finishes reliably.
- Consider caching the root candidate/scorer results when multiple recipes run
  from the same source/target/root. The first broad config was slow because each
  recipe repeated the same render/backprojection/oracle work.
- Keep generated `.zarr`, `.rrd`, and screenshots local artifacts unless the
  user explicitly asks to track them.

## Verification

- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_microset.toml --dry-run`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_microset.toml`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_target_selection.py tests/rerun_inspector/test_rollout_zarr_logger.py -q`
- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/app/panels -q`
- `cd aria_nbv && uv run ruff check aria_nbv/pose_generation/counterfactuals.py aria_nbv/pose_generation/target_counterfactuals.py aria_nbv/rollouts/dataset_writer.py tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run ruff check --select F401,TCH aria_nbv/pose_generation/counterfactuals.py aria_nbv/pose_generation/target_counterfactuals.py aria_nbv/rollouts/dataset_writer.py tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_microset.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_microset_000.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_v1_microset_000.rrd`

Rerun screenshot export could not run in this headless environment because
neither `WAYLAND_DISPLAY` nor `DISPLAY` is set.

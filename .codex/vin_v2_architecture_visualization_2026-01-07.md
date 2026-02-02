# VIN v2 architecture visualization (2026-01-07)

## Objective

Establish a **reproducible** workflow to generate **well-structured architecture diagrams** for `VinModelV2`
that can carry **shape information** (tensor/dataclass fields) and optionally **formulas** (paper-ready),
while staying **editable** (e.g., via draw.io) for final polish.

## Context snapshot

- Ran `make context` and `make context-dir-tree` (uv venv: `oracle_rri/.venv`).
- Implementation entrypoint: `oracle_rri/oracle_rri/vin/model_v2.py`.
- Typed intermediate containers used by VIN v2:
  - `oracle_rri/oracle_rri/vin/vin_v2_utils.py`: `PreparedInputs`, `PoseFeatures`, `FieldBundle`, `GlobalContext`.
  - `oracle_rri/oracle_rri/vin/types.py`: `EvlBackboneOutput`, `VinPrediction`, `VinV2ForwardDiagnostics`.

## What’s already in the repo

- Existing **Graphviz** module-level diagram:
  - Source: `docs/figures/vin_v2/vin_v2_arch.dot`
  - Rendered: `docs/figures/vin_v2/vin_v2_arch.svg`, `docs/figures/vin_v2/vin_v2_arch.png`
  - Included in the paper: `docs/typst/paper/sections/06-architecture.typ`

To regenerate (local tooling):

```bash
dot -Tsvg docs/figures/vin_v2/vin_v2_arch.dot -o docs/figures/vin_v2/vin_v2_arch.svg
dot -Tpng docs/figures/vin_v2/vin_v2_arch.dot -o docs/figures/vin_v2/vin_v2_arch.png
```

## Implemented tooling (shape-aware DOT + draw.io import)

Added a small, iteration-friendly generator:

- Script: `oracle_rri/scripts/generate_vin_v2_arch.py`
- DOT builder: `oracle_rri/oracle_rri/vin/arch_viz.py`

It runs `VinModelV2.forward_with_debug(...)` on **synthetic inputs** (no EVL checkpoint or ASE data
required) and emits:

- `*.dot` (canonical source)
- `*.svg` / `*.png` (quick inspection)
- `*.vdx` (Visio XML) which **draw.io can import as editable shapes**

Example:

```bash
oracle_rri/.venv/bin/python oracle_rri/scripts/generate_vin_v2_arch.py \
  --out-dir docs/figures/vin_v2 \
  --stem vin_v2_arch_shapes \
  --include-node-shapes \
  --enable-sem-frustum \
  --drawio
```

## VIN v2 block decomposition (as implemented)

High-level forward pass (`VinModelV2._forward_impl`):

1. **Frozen backbone (optional)**: `EvlBackbone` → `EvlBackboneOutput`
2. **Pose features**: `PoseTW` transforms into rig-ref → `PoseEncoder.encode` → `PoseFeatures(pose_enc, pose_vec, center)`
3. **Scene field**:
   - build aux channels from EVL heads: `occ_pr`, `occ_input`, `counts_norm`, `cent_pr`, `free_input`, `unknown`, `new_surface_prior`
   - project with `Conv3d(1×1×1)+GroupNorm+GELU` → `field` (`B,C,D,H,W`)
4. **Global context**:
   - compute `pos_grid` in rig-ref from `backbone_out.pts_world`
   - `PoseConditionedGlobalPool(field, pose_enc, pos_grid)` → `global_feat` (`B,N,C`)
   - optional gate by `voxel_valid_frac`
5. **Semidense view conditioning**:
   - sample semidense points (from `snippet.semidense` or cached `VinSnippetView`)
   - project into candidate cameras → projection stats (`coverage`, `empty_frac`, `semidense_candidate_vis_frac`, `depth_mean/std`)
   - optional semidense frustum MHCA → `semidense_frustum` (`B,N,C`)
   - optional PointNeXt encoder for snippet-level semidense embedding + FiLM
6. **Trajectory context (optional)**:
   - `TrajectoryEncoder.encode_poses` → per-frame encodings
   - candidate-query MH attention over trajectory tokens → `traj_ctx` (`B,N,E`)
7. **Scoring**:
   - concat: `pose_enc`, `global_feat`, semidense stats, frustum ctx, optional traj ctx, optional voxel-valid scalars
   - `head_mlp` → `CoralLayer` → `VinPrediction`

## Recommended diagram “levels”

1. **Paper-level (block diagram)**: exactly what `docs/figures/vin_v2/vin_v2_arch.dot` does today.
2. **Module-level (shape-aware)**:
   - keep nodes as **submodules** (`pose_encoder`, `field_proj`, `global_pooler`, `traj_encoder`, `head_*`, frustum attention)
   - add **edge labels** from typed containers (`PoseFeatures`, `FieldBundle`, `GlobalContext`, `VinPrediction`)
3. **Mechanism-level (selective expansion)**:
   - expand `PoseConditionedGlobalPool` (pool → tokens → LFF pos enc → MHCA → residual+MLP)
   - expand semidense frustum MHCA (token features → proj → MHCA → residual+MLP)

## Practical pipeline for “shape + formulas + editability”

Suggested approach:

1. **Get shapes from a real forward** (ground truth):
   - Run `VinModelV2.forward_with_debug(...)` on a real `VinOracleBatch`.
   - Extract `debug.*` shapes (`field_in`, `field`, `global_feat`, `semidense_proj`, `semidense_frustum`, `traj_ctx`, `logits`, …).
2. **Emit a DOT template with ports**:
   - Use Graphviz HTML-table labels and ports for dataclass-like bundles.
   - Attach edge labels from the measured shapes (or from docstring shape annotations as fallback).
3. **(Optional) Convert DOT → draw.io** for final edits + math:
   - Prefer Graphviz’ built-in `dot -Tvdx` export (the script supports `--drawio`) and import the resulting `*.vdx` in draw.io as editable shapes.
   - Enable draw.io “Mathematical Typesetting” to render LaTeX in nodes (the DOT generator keeps formulas as plain text).

References:
- Torchview (PyTorch → Graphviz): https://github.com/mert-kurttutan/torchview
- Graphviz VDX export (DOT → VDX via `dot -Tvdx`): https://graphviz.org/docs/outputs/
- draw.io MathJax support: https://www.drawio.com/blog/maths-in-diagrams
- PyTorch export dataclass/pytree registration: https://pytorch.org/docs/stable/export.html

## Notes on dataclass I/O

If a model’s `forward()` takes/returns nested dataclasses, tools usually need help:

- Prefer an explicit “flatten/unflatten” layer (PyTree) so tracing/export can see tensor leaves.
- For `torch.export`, register dataclasses as PyTrees so exported signatures can preserve structure.

---
id: 2025-12-24_vin_nbv_doc_update_2025-12-24
date: 2025-12-24
title: "Vin Nbv Doc Update 2025 12 24"
status: legacy-imported
topics: [doc, 2025, 12, 24]
source_legacy_path: ".codex/vin_nbv_doc_update_2025-12-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN docs sync (2025-12-24)

## Task

Update `docs/contents/impl/vin_nbv.qmd` so it matches the current VIN implementation in `oracle_rri/oracle_rri/vin/`,
incorporating the latest `summarize_vin` output and adding an internal VIN mermaid diagram + more math background.

## Context gathered

- `make context`, `make context-dir-tree`, `make context-qmd-tree`
- Code read:
  - `oracle_rri/oracle_rri/vin/model.py` (`VinModel`, `_build_scene_field`, frustum sampling, voxel sampling)
  - `oracle_rri/oracle_rri/vin/spherical_encoding.py` (`ShellShPoseEncoder`)
  - `oracle_rri/oracle_rri/vin/coral.py` (CORAL utilities)
  - `oracle_rri/oracle_rri/vin/rri_binning.py` (`RriOrdinalBinner`)
  - `oracle_rri/oracle_rri/lightning/lit_module.py` (`VinLightningModule.summarize_vin`)

## Key implementation snapshot (current)

- EVL feature contract used by default VIN:
  - `occ_pr: (B,1,48,48,48)`, `occ_input: (B,1,48,48,48)`, `counts: (B,48,48,48)`
  - `t_world_voxel: PoseTW(B,12)`, `voxel_extent: (B,6)`
- VIN scene field:
  - default `scene_field_channels=["occ_pr","occ_input","counts_norm"]`
  - `counts_norm_mode` default `log1p` (per-snippet normalization by max count in the volume)
  - `field_proj`: `Conv3d(C_in→16,1×1×1,bias=False) + GroupNorm + GELU`
- VIN query + scoring:
  - frustum queries use PyTorch3D `PerspectiveCameras.unproject_points(..., from_ndc=True, world_coordinates=True)`
  - `K = grid_size² × len(depths)` (defaults: `4²×4=64`)
  - token pooling is masked-mean over in-bounds samples; candidate validity uses mean(token_valid) ≥ 0.2
  - CORAL: `num_classes=15` ⇒ logits shape `(B,N,14)`, prob `(B,N,15)`, expected `(B,N)`
- Default pose encoder output dim: 64 (`u,f,r,alignment` each projected to 16 dims).

## Changes made

- Updated `docs/contents/impl/vin_nbv.qmd`:
  - Corrected counts normalization description to match `_build_scene_field` (`log1p` / per-volume max).
  - Corrected radius encoding default (`radius_log_input=False` by default).
  - Rewrote the CORAL + binning section to match `RriOrdinalBinner` + `coral.py` (removed outdated tanh/stage claims).
  - Updated “Minimal training script” section to point to the Lightning CLI (`nbv-train`) + wrappers in `oracle_rri/scripts/`.
  - Added a new VIN internal `{mermaid}` flowchart with tensor shapes (based on `summarize_vin`).
  - Added math background sections for pose encoding, scene field construction, frustum querying, voxel sampling, CORAL.
  - Added a link reference to EFM3D voxel sampling utilities (`[3]`).
- Updated `docs/references.bib`:
  - Added bib entries for LFF (arXiv:2106.02795), CORAL (arXiv:1901.07884), and relevant docs/Wikipedia pages.
  - Added Quarto citations in `vin_nbv.qmd` to these entries.

## Validation performed

- Mermaid diagram rendered with mermaid-cli:
  - `npx -y @mermaid-js/mermaid-cli -i /tmp/vin_internal.mmd -o /tmp/vin_internal.svg`
- Quarto render:
  - `quarto render docs/contents/impl/vin_nbv.qmd --to html`

## Open suggestions (not implemented)

- Consider making `EvlBackboneOutput` typing reflect that many tensors are optional depending on `features_mode` (the
  dataclass fields are currently non-optional, but the runtime may pass `None`).
- If we want true candidate filtering (instead of masking invalid candidates to zero), consider dropping invalid
  candidates from the batch and returning indices; currently `VinModel` keeps shape-stability and exposes
  `candidate_valid`.

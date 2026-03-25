---
id: 2026-01-29_paper_inconsistencies_patch_2026-01-29
date: 2026-01-29
title: "Paper Inconsistencies Patch 2026 01 29"
status: legacy-imported
topics: [inconsistencies, patch, 2026, 01, 29]
source_legacy_path: ".codex/paper_inconsistencies_patch_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Paper + Slides Consistency Patch (2026-01-29)

## Scope
Align Typst paper and slides with the current VINv3 implementation:
- remove stale claims about frustum attention and voxel gating,
- fix semidense projection weight notation,
- add missing symbol/shape definitions.

## Changes made
- Paper:
  - `docs/typst/paper/sections/06-architecture.typ` now defines reliability weights using `inv_dist_std` (σ_ρ) instead of σ_d^{-1}, and uses `clamp`.
  - `docs/typst/paper/sections/09-diagnostics.typ` and `docs/typst/paper/sections/10-discussion.typ` now clarify frustum attention is a VIN v2 ablation, not part of the v3 baseline.
  - `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ` corrected: semidense stats are fed to the head, trajectory is optional, voxel gate is off, and grid/weighting are configuration‑dependent (often 12×12). Suggestions updated to reflect current v3.
- Slides:
  - `docs/typst/slides/slides_4.typ` updated the “Mode collapse case study” slide to remove the incorrect “voxel validity feature” claim and to phrase candidate‑signal issues as ablations rather than current v3 behavior.
- Macros:
  - `docs/typst/shared/macros.typ` now includes symbols for `n_obs`, `n_obs_max`, `inv_dist_std` (σ_ρ) and its min/p95, plus `Himg`, `Wimg`, and `Gsem` shape shorthands.

## Build checks
- `typst compile typst/slides/slides_4.typ /tmp/slides_4.pdf --root .`
- `typst compile typst/paper/main.typ /tmp/main.pdf --root .`

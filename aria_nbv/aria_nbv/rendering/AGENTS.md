---
scope: module
applies_to: aria_nbv/aria_nbv/rendering/**
summary: Depth rendering, unprojection, candidate point-cloud, and rendering diagnostic guidance.
---

# Rendering Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/rendering/`. Durable rendering ownership notes live in
[README.md](README.md).

## Rules
- Treat camera, pose, depth, and point-cloud frame conventions as contracts.
- Prefer PyTorch3D, Project Aria, EFM3D, or established geometry utilities over
  new local rendering math.
- Keep rendering outputs typed and explicit about shapes, units, masks, and
  invalid-depth handling.
- Plotting and diagnostics may visualize rendering state, but must not mutate
  rendering/cache semantics.

## Verification
- Run targeted rendering and RRI tests when depth, unprojection, candidate
  point-cloud, mask, or frame semantics change.
- Add smoke tests for diagnostic figures when plotting behavior changes.

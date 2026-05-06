---
name: nbv-geometry-contracts
description: Use when ARIA-NBV work touches pose, camera, coordinate-frame, CW90, PyTorch3D projection, depth backprojection, candidate frusta, or geometry diagnostics contracts.
metadata:
  applies_to:
    - "aria_nbv/aria_nbv/pose_generation/**"
    - "aria_nbv/aria_nbv/rendering/**"
    - "aria_nbv/aria_nbv/rri_metrics/**"
    - "aria_nbv/aria_nbv/utils/data_plotting.py"
  triggers:
    - "PoseTW"
    - "CameraTW"
    - "CW90"
    - "backprojection"
  must_read:
    - "AGENTS.md"
    - "aria_nbv/AGENTS.md"
    - ".agents/memory/state/GOTCHAS.md"
  verification:
    - "cd aria_nbv && uv run pytest tests/pose_generation tests/rendering"
    - "cd aria_nbv && uv run pytest tests/rri_metrics when labels change"
    - "make context-contracts when generated contract context is needed"
---

# NBV Geometry Contracts

## When To Use

Use this skill for changes or reviews involving:

- `PoseTW`, `CameraTW`, rig/camera/world transforms, or `T_target_source` naming
- candidate poses, candidate frusta, PyTorch3D cameras, NDC, or depth maps
- CW90 corrections, gravity alignment, or display-only rotations
- depth backprojection, point-cloud construction, or visibility diagnostics

Do not use it for pure model-head, docs-only, or non-geometry app changes.

## Read First

1. `AGENTS.md`
2. `aria_nbv/AGENTS.md`
3. `.agents/memory/state/GOTCHAS.md`
4. `aria_nbv/aria_nbv/vin/AGENTS.md` when VIN batch/candidate fields are touched
5. The focused rendering or pose-generation tests for the changed path
6. `docs/_generated/context/data_contracts.md` only after `make
   context-contracts` when you need the generated contract index

## Contract Rules

- Keep semantic boundaries typed as `PoseTW` and `CameraTW`; avoid raw matrices in normal package interfaces.
- Treat `T_A_B` as transform from frame B to frame A.
- Keep CW90 and any other visual alignment correction display-only unless all affected poses, cameras, and rendered tensors are transformed in lockstep.
- Preserve PyTorch3D camera and NDC conventions in both rendering and backprojection tests.
- Document tensor shape, frame, and units whenever they are not obvious from the type.

## Verification

- `cd aria_nbv && uv run pytest tests/rendering/test_depth_backprojection_conventions.py`
- `cd aria_nbv && uv run pytest tests/rendering/test_candidate_renderer_integration.py tests/rendering/test_pytorch3d_renderer.py`
- `cd aria_nbv && uv run pytest tests/vin/test_vin_utils.py` when VIN diagnostics or batch fields are affected
- `make check-agent-memory` for guidance or memory edits

## Diagnostics Matrix

- Pose/frame edits: assert transform direction, units, and `PoseTW` batch shape.
- Camera/projection edits: assert `CameraTW` intrinsics/extrinsics and
  PyTorch3D/NDC conventions.
- Depth/backprojection edits: assert metric-depth interpretation, valid masks,
  and world-frame point bounds.
- Candidate-frustum edits: assert display-only CW90 corrections do not mutate
  training, rendering, or store semantics.
- Streamlit diagnostics: verify figures are display transforms only and do not
  feed back into model or oracle data.

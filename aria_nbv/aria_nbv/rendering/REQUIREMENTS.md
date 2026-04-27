# Rendering Mojo Requirements

## Contract

- Rendering must expose `DepthRendererBackend` and `PointCloudBackend` as
  `StrEnum`s.
- `CandidateDepthRendererConfig` must become backend-driven instead of being
  hard-wired to the PyTorch3D config shape.
- `CandidateDepths` remains the canonical depth payload and must keep these
  fields stable:
  - `depths`
  - `depths_valid_mask`
  - `poses`
  - `reference_pose`
  - `candidate_indices`
  - `camera`
- Any PyTorch3D-specific camera batch becomes optional backend state.

## Semantics

- The Mojo depth renderer must compute closest-hit metric depth from mesh
  triangles using `CameraTW` and `PoseTW` inputs.
- The Mojo point-cloud builder must fuse pixel subsampling, validity filtering,
  unprojection, compaction, and occupancy-bounds reduction.
- The point-cloud stage must not require `PerspectiveCameras` in its canonical
  contract.

## Numeric Rules

- Depth valid masks must match baseline exactly.
- Depth values and backprojected points must satisfy `atol=1e-4`, `rtol=1e-4`.
- Occupancy bounds must satisfy `atol=1e-4`, `rtol=1e-4`.

## Required Tests

- Backend enum/default contract tests.
- Synthetic renderer parity on a simple mesh.
- Synthetic point-cloud parity for backprojection and bounds.
- Real-data oracle-render parity for at least one ASE snippet when Mojo is
  available locally.

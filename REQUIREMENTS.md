# Apple-Silicon Mojo Oracle Backend Requirements

This file defines the strict repository-level requirements for the Apple-Silicon
Mojo acceleration work on the oracle RRI pipeline.

## Scope

- The parity baseline is the current checked-out PyTorch3D oracle path running on
  CPU.
- Python remains the orchestration layer.
- PyTorch3D remains the default backend until the Mojo path passes the parity and
  benchmark gates below.

## Required Toolchain

- Apple Silicon macOS host.
- Repo-local Mojo install under `.mojo-venv/` or a valid
  `ARIA_NBV_MOJO_SITE_PACKAGES` pointing at the matching site-packages directory.
- PyTorch and PyTorch3D available in the `aria_nbv/.venv` environment.

## Required Backend Surfaces

- `CollisionBackend` must expose `mojo`.
- Rendering must expose a `DepthRendererBackend` and a `PointCloudBackend`.
- RRI scoring must expose an `OracleDistanceBackend`.
- Each backend enum must be a `StrEnum`.

## Default Behavior

- All new backend seams default to the existing PyTorch3D path.
- Mojo is opt-in until the parity suite passes.
- PyTorch3D-specific runtime objects may remain available as optional fields, but
  they must not be the only canonical contract.

## Required Kernel Replacements

- Candidate collision distance and path-collision kernels.
- Candidate depth rendering.
- Depth backprojection, compaction, and occupancy-bounds reduction.
- Point-to-mesh and mesh-to-point distance kernels used by oracle RRI.

## Parity Gates

- Candidate validity masks: exact match.
- Candidate ordering / candidate indices: exact match.
- Depth valid masks: exact match.
- Depth values, backprojected points, occupancy bounds, distance components, and
  final RRI: `atol=1e-4`, `rtol=1e-4`.
- Final candidate ranking by RRI: exact match.

## Verification Gates

- `ruff format` on touched Python files.
- `ruff check` on touched Python files.
- Targeted `pytest` for pose generation, rendering, RRI metrics, and oracle
  integration parity.
- `make check-agent-memory` when canonical state or repo guidance changes.

## Documentation Gates

- Update the implementation docs under `docs/contents/impl/`.
- Update `.agents/memory/state/PROJECT_STATE.md` when the backend surface or
  workflow changes materially.
- Keep the Mermaid architecture diagram in sync with the active staged design.

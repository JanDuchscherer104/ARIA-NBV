# Pose Generation Mojo Requirements

## Contract

- `CollisionBackend` must be a `StrEnum` and include `pytorch3d`, `pyembree`,
  `trimesh`, and `mojo`.
- `CandidateContext` must support backend-local runtime caches so mesh-derived
  auxiliary buffers can be reused across rules.

## Semantics

- The default collision backend remains the current non-Mojo path.
- `MinDistanceToMeshRule` may switch to Mojo when `CollisionBackend.MOJO` is
  selected.
- `PathCollisionRule` must preserve the exact rejection semantics of the baseline.
  A temporary fallback to the established ray engine is allowed while the custom
  Mojo kernel is being verified.

## Numeric Rules

- Mesh vertices are `float32`.
- Mesh faces are `int64`.
- Candidate positions and distances remain in metres.
- Any Mojo distance result must match the baseline rule mask exactly.

## Required Tests

- Enum/default contract test.
- Synthetic parity test for `MinDistanceToMeshRule`.
- Synthetic parity test for `PathCollisionRule`.
- End-to-end candidate-generation parity on a deterministic fixture.

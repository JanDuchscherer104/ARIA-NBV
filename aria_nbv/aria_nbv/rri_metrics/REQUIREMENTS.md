# RRI Metrics Mojo Requirements

## Contract

- `OracleDistanceBackend` must be a `StrEnum`.
- `OracleRRIConfig` must expose backend selection while defaulting to the current
  PyTorch3D path.
- `RriResult` semantics must remain unchanged.

## Semantics

- AABB crop semantics in `OracleRRI` must remain unchanged.
- Accuracy is point-to-mesh distance.
- Completeness is mesh-to-point distance.
- Bidirectional distance is the sum of the two directional means.
- RRI remains `(d_before - d_after) / clamp_min(d_before, 1e-12)`.

## Numeric Rules

- Mojo point-to-mesh and mesh-to-point components must satisfy
  `atol=1e-4`, `rtol=1e-4` against the baseline.
- Empty candidate clouds must still yield `RRI == 0` and unchanged before/after
  distance components.

## Required Tests

- Backend enum/default contract tests.
- Unit parity tests on deterministic meshes and point sets.
- Oracle scorer parity tests with empty and non-empty candidate point clouds.
- End-to-end parity through `OracleRriLabeler` when Mojo is available.

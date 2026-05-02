# Rerun Python Patterns For ARIA-NBV

Use this reference when writing or reviewing Python `rerun-sdk` calls in
ARIA-NBV.

## Recording And Sinks

- Initialize exactly once per inspector run.
- Open the sink before the first data log: `save`, `spawn`, or `connect_grpc`.
- Use stable `application_id` and `recording_id` for comparable `.rrd` artifacts.
- Prefer `--save` for smoke and review artifacts; use `spawn` for interactive
  debugging and `connect` only when a viewer/server is already running.
- Keep `.rrd` artifacts outside training stores, usually under `.artifacts/rerun/`.

## Entity Layout

Use stable, low-cardinality paths:

```text
world
world/semidense
world/reference/pose
world/candidates/frusta_all
world/candidates/frusta_invalid
world/candidates/frusta_top_oracle
world/candidates/centers
world/mesh
world/gt/obbs
world/detected/obbs
world/trajectory/rig
frames/candidate_depths/<candidate_id>/image
frames/rgb/<frame_id>/image
frames/depth/<frame_id>/image
metadata/sample
```

Batch repeated geometry where possible. Avoid one entity per candidate unless
isolating a selected or failed candidate is the point of the recording.

## Coordinates And Transforms

- Log `rr.ViewCoordinates.RIGHT_HAND_Z_UP` at `/` or `world` for ARIA world
  diagnostics unless another basis is explicitly documented.
- Treat candidate `PoseTW` as `T_world_cam`.
- Treat reference trajectory `PoseTW` as `T_world_rig`.
- If logging native transforms, make relation semantics explicit. For a
  `T_world_child` pose, use the relation that describes parent-from-child, or
  invert before using child-from-parent.
- Keep manual frusta in world coordinates when they are already transformed.
- Keep camera-local depth pointmaps under a posed camera entity; keep world-space
  semidense/current/fused points under `world`.

## Cameras, RGB, And Depth

- `rr.Pinhole.resolution` is `[width, height]`.
- PyTorch3D `PerspectiveCameras.image_size` is `(height, width)` in this repo;
  convert deliberately when building Rerun pinholes.
- Use `camera_xyz=rr.ViewCoordinates.RDF` for conventional pinhole image
  entities unless a different basis is tested and documented.
- Log `Pinhole`, RGB `Image`, and metric `DepthImage(..., meter=1.0)` on the
  same camera/image entity when depth is meant to be interpreted spatially.
- If a depth tensor is logged only as a 2D diagnostic image, name it as a debug
  layer and do not imply it validates 3D projection.
- Do not log pseudo-colored point maps as raw RGB or metric depth.

## Candidate And RRI Layers

- Preserve candidate ordering across poses, cameras, RRI, validity, labels, and
  colors.
- Use `candidate_count` as the valid prefix width exactly; `candidate_count=0`
  means no candidates.
- An all-invalid validity mask means no top-oracle candidate.
- Invalid candidates are constraints, not low-RRI candidates; display them as a
  separate layer or color override.
- Store RRI, rank, validity, and candidate id in labels or metadata, not only in
  color.

## Blueprints

- Use blueprints only for viewer layout and convenience.
- Do not encode scientific facts in blueprints.
- Prefer a 3D scene rooted at `world`, optional 2D views for RGB/depth, and a
  metadata text view for inventory/config details.

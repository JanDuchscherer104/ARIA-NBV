# Context7 Queries

Use Context7 library id `/rerun-io/rerun`.

Start with one narrow query and only broaden if results are shallow. If the
repo is pinned to a specific `rerun-sdk` version, also inspect local signatures:

```bash
cd aria_nbv
uv run python - <<'PY'
import inspect, rerun as rr
for name in ["init", "save", "spawn", "connect_grpc", "log"]:
    print(name, inspect.signature(getattr(rr, name)))
for name in ["Transform3D", "Pinhole", "DepthImage", "LineStrips3D", "Points3D"]:
    print(name, inspect.signature(getattr(rr, name)))
PY
```

## Core Logging

| Goal | Query |
| --- | --- |
| Recording lifecycle | `Python init save spawn connect_grpc RecordingStream recording_id` |
| Explicit sinks | `FileSink GrpcSink set_sinks Python` |
| Entity logging | `rr.log entity path static Python` |
| Timelines | `set_time sequence timeline Python` |
| Saved recordings | `save rrd recording Python` |

## Frames And Cameras

| Goal | Query |
| --- | --- |
| Transform relation | `Transform3D TransformRelation ParentFromChild ChildFromParent Python` |
| World basis | `ViewCoordinates RIGHT_HAND_Z_UP RIGHT_HAND_Y_UP Python` |
| Pinhole camera | `Pinhole image_from_camera resolution camera_xyz Python` |
| Resolution semantics | `Pinhole resolution width height Python` |
| Metric depth | `DepthImage meter Pinhole Python` |
| Camera image tree | `Pinhole Image DepthImage same entity Python` |

## Geometry

| Goal | Query |
| --- | --- |
| Point clouds | `Points3D colors radii labels Python` |
| Frusta/lines | `LineStrips3D labels radii Python` |
| Meshes | `Mesh3D triangle_indices vertex_positions Python` |
| Boxes/OBBs | `Boxes3D InstancePoses3D Python` |
| Blueprints | `Spatial3DView Spatial2DView Blueprint Python` |

## Example Queries

| Need | Query |
| --- | --- |
| RGB-D baseline | `RGBD example Pinhole DepthImage Python` |
| SLAM-style layout | `DROID example depth pinhole blueprint Python` |
| Transform trees | `ROS TF example transform coordinate frame Python` |
| Mixed RGB/depth/mesh | `ARKit scenes example depth mesh pinhole Python` |
| Multi-camera layout | `nuScenes example pinhole blueprint Python` |

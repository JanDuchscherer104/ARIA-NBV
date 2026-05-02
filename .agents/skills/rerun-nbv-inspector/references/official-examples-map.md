# Official Rerun Examples Map

Prefer official docs and examples over recollection. Use direct example pages
first, then Context7 or GitHub source when implementation details matter.

## Start Here

- Examples index: `https://rerun.io/examples`
- Python SDK reference: `https://ref.rerun.io/docs/python/stable/common/`
- Context7 library id: `/rerun-io/rerun`

## Example Selection

### RGBD

- Page: `https://rerun.io/examples/robotics/rgbd`
- Source: `https://github.com/rerun-io/rerun/tree/docs-latest/examples/python/rgbd`
- Use for: correct `Pinhole` + `DepthImage`, `resolution=[width, height]`,
  metric depth scaling, and camera/image/depth entity layout.
- Query: `RGBD example Pinhole DepthImage Python`

### Live depth sensor

- Page: `https://rerun.io/examples/robotics/live_depth_sensor`
- Use for: live RGB/depth camera intrinsics and metric depth conventions.
- Query: `Live depth sensor example depth Pinhole Python`

### DROID

- Page: `https://rerun.io/examples/robotics/droid`
- Use for: SLAM-style dense geometry, keyframes, pinhole cameras, trajectories,
  and blueprints.
- Query: `DROID example depth pinhole blueprint Python`

### ROS TF

- Page: `https://rerun.io/examples/robotics/ros_tf`
- Use for: transform tree reasoning, parent/child frames, and timeline
  debugging when geometry lands in the wrong place.
- Query: `ROS TF example Transform3D coordinate frame Python`

### ARKit scenes

- Page: `https://rerun.io/examples/spatial-computing/arkit_scenes`
- Use for: mixed RGB, depth, mesh, OBB-like annotations, camera pose, and
  spatial-computing scene layout.
- Query: `ARKit scenes example depth mesh pinhole Python`

### nuScenes

- Page: `https://rerun.io/examples/robotics/nuscenes`
- Source: `https://github.com/rerun-io/rerun/tree/docs-latest/examples/python/nuscenes_dataset`
- Use for: multi-camera layout, static sensor calibration, transforms, and
  blueprint organization.
- Query: `nuScenes example pinhole Transform3D blueprint Python`

### Objectron

- Page: `https://rerun.io/examples/spatial-computing/objectron`
- Use for: compact camera-centric 2D/3D overlays and pinhole projections.
- Query: `Objectron example pinhole camera_xyz Python`

## Heuristic

- Start with RGBD for depth/camera questions.
- Use ROS TF for transform-relation suspicion.
- Use DROID for SLAM/reconstruction-style layout.
- Use ARKit scenes when combining mesh, depth, cameras, and boxes.
- Use nuScenes when multiple cameras or static calibration dominate.

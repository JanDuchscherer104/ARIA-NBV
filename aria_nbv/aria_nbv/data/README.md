# ASE Data Handling

Typed, IDE-friendly wrappers around ATEK WebDatasets that pair ASE snippets with ground-truth meshes and expose EFM3D-compatible helpers.

## Core Types

- **`CameraStream`** – Typed view of `MultiFrameCameraData`.
  - `images`: `[frames, channels, height, width]` in the camera frame.
  - `projection_params`: `[frames, num_params]` intrinsics.
  - `t_device_camera`: `[frames, 3, 4]` pose from camera → device (`T_device_camera`).
  - `capture_timestamps_ns`, `frame_ids`, `exposure_durations_s`, `gains`, `camera_valid_radius`.
  - `to_camera_tw()` → `CameraTW` with per-frame calibration.
- **`Trajectory`** – Device poses in world coordinates.
  - `ts_world_device`: `[frames, 3, 4]` (`T_world_device`), `gravity_in_world`, `capture_timestamps_ns`.
  - `as_pose_tw()` → `PoseTW` batch.
- **`SemiDensePoints`** – Semi-dense SLAM observations.
  - `points_world`: `List[Tensor [points, 3]]` per frame, plus std/volume metadata.
  - `stacked()` → `[frames, points, 3]`.
- **`AtekSnippet`** – Typed + raw ATEK sample (`cameras`, `trajectory`, `semidense`, `gt_data`, `raw`, `flat`).
- **`ASESample`** – Per-snippet container with `scene_id`, `snippet_id`, `atek`, optional `gt_mesh`.
  - `has_mesh`, `has_rgb`, `has_slam_points`, `has_depth`.
  - `to_flatten_dict()` for round-tripping to ATEK.
  - `to_efm_dict()` remaps keys via `EfmModelAdaptor` for EFM3D ingestion.

## Dataset

`ASEDataset` is an `IterableDataset[ASESample]` that streams ATEK shards and attaches meshes.

```python
from torch.utils.data import DataLoader
from aria_nbv.data_handling import ASEDatasetConfig, ase_collate

config = ASEDatasetConfig(
    scene_ids=["81022", "81048"],       # optional; auto-resolves tars and meshes
    atek_variant="efm",                 # subdir under .data/ase_atek
    batch_size=None,                    # let DataLoader batch
    load_meshes=True,
    require_mesh=False,
)

dataset = config.setup_target()
loader = DataLoader(dataset, batch_size=4, collate_fn=ase_collate)

batch = next(iter(loader))
print(batch["scene_id"])     # ['81022', '81048', ...]
print(batch["efm"][0].keys())  # EFM3D key set (rgb/img, pose/t_world_rig, points/p3s_world, ...)
```

### Config Fields

- `tar_urls`: Tar file paths or glob patterns (autofilled from `scene_ids` if provided).
- `scene_to_mesh`: `scene_id -> Path` (autofilled when `scene_ids`+`load_meshes` are set).
- `atek_variant`: Subdirectory under `.data/ase_atek/` (default: `efm`).
- `batch_size`: WebDataset batch size (prefer `None` + DataLoader batching).
- `shuffle`, `repeat`: Passed to `load_atek_wds_dataset`.
- `mesh_simplify_ratio`: Optional decimation `0-1`.
- `cache_meshes`: Cache loaded meshes in memory.

### Collation

Use `ase_collate` with PyTorch `DataLoader` for ergonomic batching:

```python
loader = DataLoader(dataset, batch_size=8, collate_fn=ase_collate)
for batch in loader:
    rgb = batch["atek"][0].camera_rgb.images  # typed access
    meshes = batch["gt_mesh"]                  # list[Trimesh | None]
```

## Key Remapping to EFM3D

`ASESample.to_efm_dict()` uses `EfmModelAdaptor.get_dict_key_mapping_all()` to translate ATEK keys:

- `mtd#ts_world_device` → `pose/t_world_rig`
- `mtd#capture_timestamps_ns` → `pose/time_ns`
- `mfcd#camera-rgb+images` → `rgb/img`
- `mfcd#camera-rgb+t_device_camera` → `rgb/calib/t_device_camera`
- `msdpd#points_world` → `points/p3s_world`
- Plus camera calibration keys per stream (rgb/slaml/slamr)

The returned dict stays compatible with downstream EFM3D utilities.

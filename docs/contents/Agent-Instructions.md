# Agent Instructions: Oracle RRI Pipeline

## Project Snapshot

- **Goal**: train an RRI-based NBV policy that evaluates candidate poses via Relative Reconstruction Improvement (RRI) instead of coverage proxies.
- **Dataset**: Aria Synthetic Environments (ASE) – semi-dense SLAM point clouds, RGB/SLAM fisheyes, GT meshes.
- **Current focus (Phase 2)**: generate oracle RRI labels by comparing partial reconstructions against ground-truth meshes.
- **Key references**: `docs/contents/theory/rri_theory.qmd`, `docs/contents/impl/rri_computation.qmd`, `docs/contents/impl/atek_implementation.qmd`, `docs/contents/impl/efm3d_implementation.qmd`.

## Core Metrics

### Relative Reconstruction Improvement
$$
\text{RRI}(q)=\frac{\text{CD}(P_t, M_{GT})-\text{CD}(P_{t\cup q}, M_{GT})}{\text{CD}(P_t, M_{GT})}
$$
Higher values indicate a more useful candidate pose \(q\). The denominator enforces scale invariance.

### Bidirectional Chamfer Distance
$$
\text{CD}(P,M)=\mathbb{E}_{p\in P}\min_{m\in M}\|p-m\|+\mathbb{E}_{m\in M}\min_{p\in P}\|m-p\|
$$
The first term (accuracy) penalises hallucinated geometry; the second (completeness) captures missing surfaces. **Both** terms must be evaluated with matched point-cloud densities.

## Libraries & Canonical Usage

### Project Aria Tools (`external/projectaria_tools`)
Handles fisheye cameras, ASE CSV exports, and SE(3) transforms.

```python
from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3

camera_calib = device_calibration.get_camera_calib(CameraId.RGB)
ray_cam = camera_calib.unproject([u, v])       # Kannala–Brandt fisheye
depth_m = depth_mm / 1000.0
point_cam = depth_m * ray_cam
T_world_camera: SE3 = T_world_device @ camera_calib.get_transform_device_camera()
point_world = T_world_camera * point_cam
```

ASE helpers (CSV exports) live in `projectaria_tools.projects.ase.readers`.

> **Never** use `open3d.geometry.create_from_depth_image` with Aria fisheye data; it assumes a pinhole model.

### ATEK (`external/ATEK`)
Provides production mesh metrics.

```python
from atek.evaluation.surface_reconstruction.surface_reconstruction_metrics import evaluate_single_mesh_pair
from atek.evaluation.surface_reconstruction.surface_reconstruction_utils import compute_pts_to_mesh_dist

# Full evaluation (file paths)
metrics, accuracy, completeness = evaluate_single_mesh_pair(pred_mesh, gt_mesh, sample_num=10_000, step=50_000)

# Direct distances (tensors)
dists = compute_pts_to_mesh_dist(points, faces, vertices, step=50_000)
```

`compute_pts_to_mesh_dist` relies on `rtree` for best performance; install it or expect a slow fallback.

### EFM3D (`external/efm3d`, editable install)
Vectorised ray and point-cloud utilities implemented in PyTorch.

```python
from efm3d.utils.ray import ray_grid, transform_rays
from efm3d.utils.pointcloud import get_points_world, collapse_pointcloud_time
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW

camera = CameraTW.from_parameters(width=torch.tensor([1408]), height=torch.tensor([1408]), ...)
rays_rig = ray_grid(camera)              # (H, W, 6): [origin_xyz, direction_xyz]
rays_world = transform_rays(rays_rig, pose_snippet)
points_world, dist_std = get_points_world(batch, batch_idx=0)
```

See `docs/contents/impl/efm3d_implementation.qmd` for a function survey.

## Data Layout & Download Targets

- **WebDataset shards**: `.data/ase_atek_eval/<scene>/shards-000X.tar`
- **GT meshes**: `.data/ase_meshes/scene_ply_<id>.ply`
- **Raw ASE chunks** (optional depth/RGB): `.data/ase_raw/`

```bash
# ATEK-preprocessed WebDataset shards
python3 external/ATEK/tools/atek_wds_data_downloader.py \
  --config-name efm \
  --input-json-path .data/aria_download_urls/AriaSyntheticEnvironment_ATEK_download_urls.json \
  --output-folder-path .data/ase_atek_eval \
  --max-num-sequences 2

# GT meshes (100 eval scenes)
python3 external/ATEK/tools/ase_mesh_downloader.py \
  --input-json .data/aria_download_urls/ase_mesh_download_urls.json \
  --output-dir .data/ase_meshes

# Raw ASE (optional RGB/depth/VRS)
python3 external/projectaria_tools/projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py \
  --set train --scene-ids 560-569 \
  --cdn-file .data/aria_download_urls/aria_synthetic_environments_dataset_download_urls.json \
  --output-dir .data/ase_raw
```

`webdataset` decodes each shard; `.pth` entries such as `msdpd#points_world+stacked.pth` and `msdpd#points_world_lengths.pth` are loaded with `torch.load` and wrapped as `OracleRRISnippet` objects.

## Implementation Principles

- **Consistent sampling**: before comparing \(P_t\) and \(P_{t\cup q}\) always harmonise densities:

  ```python
  def voxel_average(points: torch.Tensor, voxel_size: float = 0.01) -> torch.Tensor:
      if points.numel() == 0:
          return points
      coords = torch.floor(points / voxel_size).to(torch.int64)
      unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
      counts = torch.zeros(unique.shape[0], device=points.device, dtype=points.dtype)
      sums = torch.zeros(unique.shape[0], 3, device=points.device, dtype=points.dtype)
      sums.index_add_(0, inverse, points)
      counts.index_add_(0, inverse, torch.ones_like(counts))
      return sums / counts.unsqueeze(1)
  ```

- **Accuracy vs completeness**: use `compute_pts_to_mesh_dist` for the \(P\to M\) term; run a KD-tree or another point-to-point query for \(M\to P\).
- **Coordinate frames**: world ← device ← camera. Keep all SE(3) transforms explicit; depth values are in millimetres.
- **External repos**: `external/ATEK` and `external/projectaria_tools` are vendored submodules—do not edit in-place. `external/efm3d` is editable.

## Quick Commands

```bash
conda activate aria-nbv
cd oracle_rri && uv pip install -e ".[dev, notebook]"
pytest oracle_rri_pkg/tests/   # unit tests
```

## Contact Points

- Documentation: `docs/contents/todos.qmd` (tasks), `docs/contents/questions.qmd` (open research), `docs/contents/roadmap.qmd` (phase plan).
- Notebooks: exploratory work lives under `notebooks/ase_*` and `notebooks/scalable_rri_implementation.ipynb`.

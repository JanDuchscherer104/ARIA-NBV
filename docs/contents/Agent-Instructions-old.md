# Agent Instructions: NBV Planning with Foundation Models

You are an expert AI research assistant developing a Next-Best-View (NBV) planning system that uses **Relative Reconstruction Improvement (RRI)** to directly optimize reconstruction quality, leveraging **egocentric foundation models (EFM3D)** for 3D spatial understanding in complex indoor scenes.

---

## Project Goal

**Build RRI-based NBV system** that:
- Directly optimizes reconstruction quality (not coverage proxies)
- Uses EVL foundation features as state embeddings for RRI prediction
- Trains on ASE dataset (100K scenes, GT meshes, semi-dense point clouds)

**Current Phase**: Oracle RRI implementation (Phase 2) - computing ground-truth RRI scores for training data generation.

---

## Core Concepts

### RRI (Relative Reconstruction Improvement)
$$\text{RRI}(q) = \frac{\text{CD}(P_t, M_{GT}) - \text{CD}(P_{t \cup q}, M_{GT})}{\text{CD}(P_t, M_{GT})}$$

**Components**:
- $q \in SE(3)$: Candidate viewpoint (position + orientation)
- $P_t$: Current point cloud reconstruction
- $P_{t \cup q}$: Updated reconstruction after capturing view $q$
- $M_{GT}$: Ground truth mesh
- $\text{CD}$: Chamfer Distance (accuracy + completeness)

**Properties**: Range [0,1], higher = better viewpoint. **Critical challenge**: Must ensure consistent point cloud sampling for valid CD comparison.

### Chamfer Distance (Bidirectional Metric)
$$\text{CD}(P, M) = \underbrace{\text{mean}_{p \in P} \min_{m \in M} \|p-m\|}_{\text{Accuracy (P→M)}} + \underbrace{\text{mean}_{m \in M} \min_{p \in P} \|m-p\|}_{\text{Completeness (M→P)}}$$

**Why bidirectional**: Accuracy detects over-reconstruction, completeness detects under-reconstruction.

---

## Technology Stack & APIs

### 1. ProjectAria Tools (Camera Models & Data Access)

**Critical**: Aria uses **fisheye/Kannala-Brandt** cameras (NOT pinhole). Always use ProjectAria's unprojection.

```python
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.mps import get_eyegaze_point_at_depth

# Camera unprojection (handles fisheye distortion)
camera_calib = device_calibration.get_camera_calib(CameraId.RGB)
ray = camera_calib.unproject([u, v])  # Pixel → 3D ray direction
p_camera = depth_m * ray              # Scale by depth

# SE3 transforms (world ← device ← camera)
T_world_camera = T_world_device @ T_device_camera
p_world = T_world_camera @ p_camera
```

**⚠️ Never use**: `open3d.geometry.create_from_depth_image()` - assumes pinhole, causes distortion!

**Key modules**:
- `calibration`: Camera intrinsics, distortion models
- `data_provider`: VRS file reading, sensor streams
- `sophus`: SE(3) poses, transformations
- `projects.ase`: ASE dataset readers, scene interpreters

**Path**: `external/projectaria_tools/` (NOT editable - git submodule)

### 2. ATEK (Mesh Evaluation & Metrics)

**Production-ready surface reconstruction evaluation.**

```python
from atek.evaluation.surface_reconstruction.surface_reconstruction_metrics import evaluate_single_mesh_pair
from atek.evaluation.surface_reconstruction.surface_reconstruction_utils import compute_pts_to_mesh_dist

# Complete mesh evaluation pipeline
metrics, accuracy, completeness = evaluate_single_mesh_pair(
    pred_mesh_filename="prediction.ply",
    gt_mesh_filename="ground_truth.ply",
    sample_num=10000,      # Points sampled from each mesh
    step=50000,            # Batch size for memory management
    threshold=0.05,        # 5cm threshold for precision/recall
    rnd_seed=42
)
# Returns: accuracy_mean, completeness_mean, prec@0.05, recall@0.05, fscore@0.05

# Efficient point-to-mesh distance (batched, handles millions of points)
distances = compute_pts_to_mesh_dist(
    points,      # (N, 3) query points
    faces,       # (F, 3) mesh face indices
    vertices,    # (V, 3) mesh vertices
    step=50000   # Batch size
)  # Returns: (N,) distances
```

**Key algorithms**:
- `evaluate_single_mesh_pair()`: End-to-end mesh evaluation with all metrics
- `compute_pts_to_mesh_dist()`: Batched point-to-triangle distance (barycentric projection)
- `point_to_closest_tri_dist()`: Geometric projection with slab method

**Path**: `external/ATEK/` (NOT editable - git submodule)
**Docs**: `docs/contents/impl/atek_implementation.qmd`

### 3. EFM3D (Ray Utilities & Point Cloud Ops)

**Foundation model utilities for egocentric 3D reasoning.**

```python
from efm3d.utils.ray import ray_grid, transform_rays, sample_depths_in_grid
from efm3d.utils.pointcloud import get_points_world, collapse_pointcloud_time, pointcloud_to_voxel_ids
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW

# Vectorized ray generation (NO manual loops!)
camera = CameraTW(fx, fy, cx, cy, width, height)
rays_camera = ray_grid(camera)  # Shape: (H, W, 6) [origin_xyz, direction_xyz]

# Transform rays between coordinate frames
pose = PoseTW(R=rotation_matrix, t=translation_vector)
rays_world = transform_rays(rays_camera, pose.matrix())

# Point cloud from depth maps
pc_world = get_points_world(depth_map, camera, pose)  # Handles camera models

# Temporal merging (removes NaN, deduplicates)
pc_merged = collapse_pointcloud_time(point_clouds)  # Input: (B, T, N, 3)

# Voxelization
voxel_ids = pointcloud_to_voxel_ids(points, voxel_min, voxel_max, grid_dims)
```

**Key utilities**:
- **Ray ops**: `ray_grid()`, `transform_rays()`, `ray_obb_intersection()`, `sample_depths_in_grid()`
- **Point clouds**: `get_points_world()`, `collapse_pointcloud_time()`, `pointcloud_to_voxel_ids()`
- **Mesh eval**: `eval_mesh_to_mesh()` (extends ATEK with visualizations)

**Path**: `external/efm3d/` (Installed editable - can modify if needed)
**Docs**: `docs/contents/impl/efm3d_implementation.qmd`, `docs/contents/impl/rri_computation.qmd`

---

## Repository Structure

```
/home/jandu/repos/NBV/
├── oracle_rri/                    # Main implementation package
│   ├── pyproject.toml            # Dependencies and build config
│   ├── environment.yml           # Conda environment specification
│   ├── oracle_rri/               # Source code
│   └── tests/                    # Unit tests
├── notebooks/                     # Jupyter exploration notebooks
│   ├── ase_exploration.ipynb     # ASE data analysis (point cloud fixes)
│   ├── ase_atek_exploration.ipynb
│   ├── scalable_rri_implementation.ipynb
│   └── inference.ipynb
├── docs/                          # Project documentation (Quarto)
│   ├── contents/
│   │   ├── theory/               # RRI, surface metrics, NBV background
│   │   ├── impl/                 # Implementation guides (ATEK, EFM3D, RRI)
│   │   ├── literature/           # Paper reviews (VIN-NBV, GenNBV, EFM3D, SceneScript)
│   │   ├── roadmap.qmd          # Development phases and timeline
│   │   ├── questions.qmd        # Open research questions
│   │   ├── todos.qmd            # Action items and current tasks
│   │   └── resources.qmd        # External links and tools
│   └── index.qmd                # Project overview and abstract
├── external/
│   ├── ATEK/                    # NOT installed editable
│   ├── projectaria_tools/       # NOT installed editable
│   ├── efm3d/                   # Installed editable
│   └── scenescript/
└── tools/                        # Utility scripts
    └── ase_coordinated_downloader.py
```

### Documentation Navigation

- **Project State**: `roadmap.qmd`, `questions.qmd`, `todos.qmd`
- **Theory**: `theory/rri_theory.qmd`, `theory/surface_metrics.qmd`
- **Dataset**: `ase_dataset.qmd`, `resources.qmd`
- **Literature**: `literature/vin_nbv.qmd`, `literature/efm3d.qmd`, etc.
- **Implementation**: `impl/atek_implementation.qmd`, `impl/efm3d_implementation.qmd`, `impl/rri_computation.qmd`

---

## Development Guidelines

### Code Quality Rules

1. **Test-Driven Development**: Always write tests before or alongside implementation
2. **Documentation**: Update `.qmd` files when implementing new features
3. **Type Hints**: Use Python type hints for all function signatures
4. **Modularity**: Keep functions focused, avoid god classes
5. **Error Handling**: Use descriptive error messages, handle edge cases

### Working with External Packages

**CRITICAL**: ATEK and projectaria_tools are NOT installed as editable packages (git submodules only).

- **When to use tools**: Reference their implementations, copy patterns, call installed utilities
- **Do NOT modify**: Never edit files in `external/ATEK` or `external/projectaria_tools` directly
- **Import safely**: Check if packages are installed before importing
- **EFM3D exception**: This IS installed editable, can be modified if needed

### Camera Projection Workflow

**Always use ProjectAria's camera models for Aria data**:

```python
from projectaria_tools.core import calibration

# Load camera calibration
camera_calib = ... # from device_calibration.get_camera_calib(CameraId)

# Unproject pixel to 3D (accounts for fisheye distortion)
for v in range(height):
    for u in range(width):
        ray = camera_calib.unproject([u, v])  # Unit direction vector
        depth_m = depth_map[v, u] / 1000.0    # Convert mm to m
        p_camera = depth_m * ray               # 3D point in camera frame

        # Transform to world frame
        T_world_camera = T_world_device @ T_device_camera
        p_world = T_world_camera @ p_camera
```

**Never use Open3D's projection for Aria** (it assumes pinhole):
```python
# ❌ WRONG - causes distortion with Aria's fisheye cameras
o3d.geometry.PointCloud.create_from_depth_image(...)
```

### Data Management

**ASE Dataset Download**:
```bash
# 1. Download ATEK-preprocessed data
python3 external/ATEK/tools/atek_wds_data_downloader.py \
  --config-name efm \
  --input-json-path .data/aria_download_urls/AriaSyntheticEnvironment_ATEK_download_urls.json \
  --output-folder-path .data/ase_atek \
  --max-num-sequences 2

# 2. Download GT meshes (100 validation scenes)
python3 external/ATEK/tools/ase_mesh_downloader.py \
  --input-json .data/aria_download_urls/ase_mesh_download_urls.json \
  --output-dir .data/ase_meshes

# 3. Download raw ASE (for depth maps, RGB)
python3 external/projectaria_tools/projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py \
  --set train \
  --scene-ids 560-569 \
  --cdn-file .data/aria_download_urls/aria_synthetic_environments_dataset_download_urls.json \
  --output-dir .data/ase_raw
```

---

## Current Tasks and Priorities

### Immediate Action Items (from `todos.qmd`)

**HIGHEST PRIORITY**:
1. Implement `OracleRRI` class with Chamfer Distance computation
2. Implement `CandidateViewGenerator` for sampling candidate poses
3. Integrate ray casting and point cloud sampling from candidate views

**Data Management**:
- Implement coordinated ASE dataset downloader (ATEK + raw + GT meshes)
- Create metadata mapping: scene_id ↔ snippet_ids
- Setup data directory structure and caching

**Testing & Validation**:
- Validate RRI computation against theoretical expectations
- Compare with VIN-NBV approach on simple scenes
- Test memory usage and optimization strategies

### Open Research Questions (from `questions.qmd`)

1. **RRI Computation**: Which oracle RRI formulation is most predictive? How to handle point cloud sampling distribution mismatch?
2. **Model Architecture**: Should we explicitly project features into candidate view frame or use learnable positional encodings?
3. **Entity-Aware NBV**: Can we compute per-entity reconstruction completeness scores? How to weight entity importance?
4. **Action Space**: Discrete view selection (VIN-NBV) vs continuous pose regression (GenNBV)?

---

## Response Guidelines

### When Implementing Code

1. **Understand context first**: Read relevant `.qmd` documentation files before implementing
2. **Check existing implementations**: Look at ATEK/EFM3D patterns in `docs/contents/impl/`
3. **Use type hints**: All functions should have typed parameters and return values
4. **Add docstrings**: Include purpose, parameters, returns, and usage examples
5. **Update documentation**: Modify corresponding `.qmd` files when adding features

### When Debugging

1. **Check camera models**: Verify using ProjectAria's unprojection, not Open3D
2. **Validate coordinates**: Ensure SE3 transform chain is correct (world ← device ← camera)
3. **Test with simple cases**: Use single-view scenarios before multi-view fusion
4. **Visualize intermediates**: Plot point clouds, depth maps, transforms at each step
5. **Reference tutorials**: Check `external/projectaria_tools/examples/` notebooks

### When Answering Questions

1. **Be specific**: Reference file paths, function names, line numbers
2. **Show examples**: Provide code snippets from existing codebase
3. **Explain tradeoffs**: Discuss alternative approaches and their pros/cons
4. **Link documentation**: Point to relevant `.qmd` files for deeper context
5. **Acknowledge uncertainty**: If unsure, suggest experiments or references to check

---

## Examples

<example type="good_implementation">
**Task**: Implement depth map to point cloud conversion for ASE

**Approach**:
1. Read `docs/contents/theory/surface_metrics.qmd` for metric definitions
2. Check `external/projectaria_tools/examples/Gen1/python_notebooks/adt_depth_maps_to_pointcloud_tutorial.ipynb`
3. Use `camera.unproject()` for each pixel (not Open3D)
4. Apply SE3 transform chain: `T_world_camera = T_world_device @ T_device_camera`
5. Test on single frame before multi-frame fusion
6. Document approach in `docs/contents/impl/rri_computation.qmd`
</example>

<example type="good_debugging">
**Issue**: Point clouds look distorted

**Debugging steps**:
1. Check camera model: Are we using pinhole assumption? (Open3D does this)
2. Verify transform chain: Is `T_world_camera` computed correctly?
3. Validate depth units: Are we converting mm → m?
4. Test with ADT tutorial approach: Does official method work?
5. Compare with EFM3D's point cloud utilities
6. Document root cause and fix in notebook markdown cell
</example>

---

## Important Notes

- **Coordinate frames**: Always be explicit about which frame you're in (world, device, camera)
- **Units**: Depth maps are in millimeters, convert to meters for computation
- **Memory**: ASE scenes are large, use subsampling and batching strategies
- **Reproducibility**: Set random seeds, document hyperparameters
- **Version control**: Commit frequently with descriptive messages

---

## Quick Reference

### Key Files

- **Setup**: `oracle_rri/pyproject.toml`, `oracle_rri/environment.yml`
- **Current work**: `notebooks/ase_exploration.ipynb`, `notebooks/scalable_rri_implementation.ipynb`
- **Action items**: `docs/contents/todos.qmd`
- **Theory**: `docs/contents/theory/rri_theory.qmd`

### Key Commands

```bash
# Activate environment
conda activate aria-nbv

# Install package (editable)
cd oracle_rri && uv pip install --python "$(which python)" -e ".[dev, notebook]"

# Run tests
pytest oracle_rri/tests/

# Build documentation
cd docs && quarto preview
```

### Key Imports

```python
# ProjectAria
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.mps import get_eyegaze_point_at_depth

# ATEK
from atek.evaluation.surface_reconstruction import surface_reconstruction_metrics

# EFM3D
from efm3d.utils import mesh_utils, pointcloud
from efm3d.aria import camera, pose

# Standard
import torch
import pytorch3d
import trimesh
import open3d as o3d
```

---

**Remember**: You are working on cutting-edge research. Be rigorous, document thoroughly, and don't hesitate to propose novel solutions to open problems. Always prioritize correctness over speed, and clarity over cleverness.

```
├── contents
│   ├── ase_dataset.qmd
│   ├── glossary.qmd
│   ├── impl
│   │   ├── atek_implementation.qmd
│   │   ├── efm3d_implementation.qmd
│   │   ├── efm3d_symbol_index.qmd
│   │   ├── oracle_rri_class.qmd
│   │   ├── oracle_rri_impl.qmd
│   │   ├── overview.qmd
│   │   └── rri_computation.qmd
│   ├── literature
│   │   ├── efm3d.qmd
│   │   ├── gen_nbv.qmd
│   │   ├── index.qmd
│   │   ├── scene_script.qmd
│   │   └── vin_nbv.qmd
│   ├── questions.qmd
│   ├── resources.qmd
│   ├── roadmap.qmd
│   ├── setup.qmd
│   ├── theory
│   │   ├── nbv_background.qmd
│   │   ├── rri_theory.qmd
│   │   ├── semi-dense-pc.qmd
│   │   └── surface_metrics.qmd
│   └── todos.qmd
```

ATEK repository:
```
external/ATEK
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── atek
│   ├── __init__.py
│   ├── configs
│   │   └── __init__.py
│   ├── data_download
│   │   ├── __init__.py
│   │   └── atek_data_store_download.py
│   ├── data_loaders
│   │   ├── __init__.py
│   │   ├── atek_raw_dataloader_as_cubercnn.py
│   │   ├── atek_wds_dataloader.py
│   │   ├── cubercnn_model_adaptor.py
│   │   ├── sam2_model_adaptor.py
│   │   └── test
│   │       ├── __init__.py
│   │       └── atek_wds_dataloader_test.py
│   ├── data_preprocess
│   │   ├── __init__.py
│   │   ├── atek_data_sample.py
│   │   ├── atek_wds_writer.py
│   │   ├── genera_atek_preprocessor_factory.py
│   │   ├── general_atek_preprocessor.py
│   │   ├── processors
│   │   │   ├── __init__.py
│   │   │   ├── aria_camera_processor.py
│   │   │   ├── depth_image_processor.py
│   │   │   ├── efm_gt_processor.py
│   │   │   ├── mps_online_calib_processor.py
│   │   │   ├── mps_semidense_processor.py
│   │   │   ├── mps_traj_processor.py
│   │   │   ├── obb2_gt_processor.py
│   │   │   └── obb3_gt_processor.py
│   │   ├── sample_builders
│   │   │   ├── __init__.py
│   │   │   ├── atek_data_paths_provider.py
│   │   │   ├── efm_sample_builder.py
│   │   │   └── obb_sample_builder.py
│   │   ├── subsampling_lib
│   │   │   ├── __init__.py
│   │   │   └── temporal_subsampler.py
│   │   ├── test
│   │   │   ├── __init__.py
│   │   │   ├── aria_camera_processor_test.py
│   │   │   ├── atek_data_sample_test.py
│   │   │   ├── depth_image_processor_test.py
│   │   │   ├── file_io_utils_test.py
│   │   │   ├── mps_processor_test.py
│   │   │   ├── obb2_gt_processor_test.py
│   │   │   ├── obb3_gt_processor_test.py
│   │   │   └── obb_sample_builder_test.py
│   │   └── util
│   │       └── __init__.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── static_object_detection
│   │   │   ├── __init__.py
│   │   │   ├── eval_obb3.py
│   │   │   ├── eval_obb3_metrics_utils.py
│   │   │   ├── obb3_csv_io.py
│   │   │   └── static_object_detection_metrics.py
│   │   └── surface_reconstruction
│   │       ├── __init__.py
│   │       ├── surface_reconstruction_metrics.py
│   │       └── surface_reconstruction_utils.py
│   ├── util
│   │   ├── __init__.py
│   │   ├── atek_constants.py
│   │   ├── camera_calib_utils.py
│   │   ├── file_io_utils.py
│   │   ├── tensor_utils.py
│   │   └── viz_utils.py
│   └── viz
│       ├── __init__.py
│       ├── atek_visualizer.py
│       └── cubercnn_visualizer.py
├── docs
│   ├── ATEK_Data_Store.md
│   ├── Install.md
│   ├── ML_task_object_detection.md
│   ├── ML_task_surface_recon.md
│   ├── ModelAdaptors.md
│   ├── data_loading_and_inference.md
│   ├── evaluation.md
│   ├── example_cubercnn_customization.md
│   ├── example_demos.md
│   ├── example_sam2_customization.md
│   ├── example_training.md
│   ├── preprocessing.md
│   └── preprocessing_configurations.md
├── readme.md
├── setup.py
├── setup_for_pywheel.py
└── tools
    ├── ase_mesh_downloader.py
    ├── atek_wds_data_downloader.py
    ├── benchmarking_static_object_detection.py
    ├── benchmarking_surface_reconstruction.py
    ├── infer_cubercnn.py
    └── train_cubercnn.py
```

EFM3D repository:
```
external/efm3d
├── INSTALL.md
├── README.md
├── benchmark.md
├── ckpt
│   └── README.md
├── data
│   ├── README.md
│   ├── dataverse_url_parser.py
│   └── download_ase_mesh.py
├── efm3d
│   ├── __init__.py
│   ├── aria
│   │   ├── __init__.py
│   │   ├── aria_constants.py
│   │   ├── camera.py
│   │   ├── obb.py
│   │   ├── pose.py
│   │   ├── projection_utils.py
│   │   └── tensor_wrapper.py
│   ├── dataset
│   │   ├── atek_vrs_dataset.py
│   │   ├── atek_wds_dataset.py
│   │   ├── augmentation.py
│   │   ├── efm_model_adaptor.py
│   │   ├── vrs_dataset.py
│   │   └── wds_dataset.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── fuse.py
│   │   ├── model.py
│   │   ├── pipeline.py
│   │   ├── track.py
│   │   └── viz.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── dinov2_utils.py
│   │   ├── dpt.py
│   │   ├── evl.py
│   │   ├── evl_train.py
│   │   ├── image_tokenizer.py
│   │   ├── lifter.py
│   │   └── video_backbone.py
│   ├── thirdparty
│   │   ├── __init__.py
│   │   └── mmdetection3d
│   │       ├── __init__.py
│   │       ├── cuda
│   │       │   └── setup.py
│   │       └── iou3d.py
│   └── utils
│       ├── __init__.py
│       ├── common.py
│       ├── depth.py
│       ├── detection_utils.py
│       ├── evl_loss.py
│       ├── file_utils.py
│       ├── gravity.py
│       ├── image.py
│       ├── image_sampling.py
│       ├── marching_cubes.py
│       ├── mesh_utils.py
│       ├── obb_csv_writer.py
│       ├── obb_io.py
│       ├── obb_matchers.py
│       ├── obb_metrics.py
│       ├── obb_trackers.py
│       ├── obb_utils.py
│       ├── pointcloud.py
│       ├── ray.py
│       ├── reconstruction.py
│       ├── render.py
│       ├── rescale.py
│       ├── viz.py
│       ├── voxel.py
│       └── voxel_sampling.py
├── eval.py
├── infer.py
└── train.py
```

Important paths:
external/projectaria_tools/examples/Gen1/python_notebooks
├── dataprovider_quickstart_tutorial.ipynb
├── mps_quickstart_tutorial.ipynb
├── sophus_quickstart_tutorial.ipynb
└── ticsync_tutorial.ipynb

external/projectaria_tools/projects/AriaSyntheticEnvironment
├── aria_synthetic_environments_downloader.py
├── python
│   ├── CalibrationProviderPyBind.h
│   ├── TestBindings.py
│   └── bindings.cpp
└── tutorial
    ├── ase_tutorial_notebook.ipynb
    └── code_snippets
        ├── constants.py
        ├── interpreter.py
        ├── plotters.py
        └── readers.py


external/projectaria_tools/projectaria_tools
├── __init__.py
├── core
│   ├── __init__.py
│   ├── calibration.py
│   ├── data_provider.py
│   ├── gen2_mp_csv_exporter.py
│   ├── image.py
│   ├── mps
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── sensor_data.py
│   ├── sophus.py
│   ├── stream_id.py
│   ├── vrs.py
│   └── xprs.py
├── projects
│   ├── __init__.py
│   ├── adt
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── aea
│   │   └── __init__.py
│   ├── ase
│   │   ├── __init__.py
│   │   ├── interpreter.py
│   │   └── readers.py
│   └── dtc_objects
│       ├── __init__.py
│       ├── downloader_lib.py
│       └── downloader_main.py
├── tools
│   ├── __init__.py
│   ├── aria_rerun_viewer
│   │   ├── __init__.py
│   │   ├── aria_data_plotter.py
│   │   └── aria_rerun_viewer.py
│   ├── dataset_downloader
│   │   ├── __init__.py
│   │   ├── dataset_download_status_manager.py
│   │   ├── dataset_downloader.py
│   │   ├── dataset_downloader_main.py
│   │   └── dataset_downloader_utils.py
│   ├── gen2_mp_csv_exporter
│   │   ├── __init__.py
│   │   └── run_gen2_mp_csv_exporter.py
│   ├── viewer_mps
│   │   ├── __init__.py
│   │   ├── rerun_viewer_mps.py
│   │   └── viewer_mps.py
│   ├── viewer_projects
│   │   ├── viewer_projects_adt.py
│   │   ├── viewer_projects_aea.py
│   │   └── viewer_projects_ase.py
│   └── vrs_to_mp4
│       ├── __init__.py
│       ├── vrs_to_mp4.py
│       └── vrs_to_mp4_utils.py
└── utils
    ├── __init__.py
    ├── calibration_utils.py
    └── rerun_helpers.py
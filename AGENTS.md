# NBV - Next-Best-View Planning with Egocentric Foundation Models

This document orients AI coding agents and contributors working on the **NBV – Next-Best-View Planning with Foundation Models** project. It explains the project’s goals, core architecture, environment, and the workflow and style rules you must follow before making changes.

---

## 1. Mission & Scope

We are building an **active Next-Best-View planning system** for complex indoor scenes, using:

- **Relative Reconstruction Improvement (RRI)** as the core metric for viewpoint selection.
- **Pre-trained egocentric foundation models (EFM3D/EVL)** for rich 3D spatial understanding.
- **Aria Synthetic Environments (ASE)** data with GT meshes, semi-dense SLAM points, and depth.

### Key Components

- **Oracle RRI Computation**: Ground-truth RRI calculation using ASE dataset's GT meshes, semi-dense SLAM point clouds, and depth maps. Serves as training labels for the learned RRI predictor.
- **Candidate View Generation**: Generates SE(3) camera poses around the current trajectory for NBV evaluation.
- **RRI Predictor Head**: Lightweight network trained on top of frozen EFM3D backbone to predict reconstruction improvement for candidate views.
- **Entity-Aware NBV**: Leverages EVL's OBB detection for task-specific view suggestions and weighted RRI computation.

### Technology Stack

- **Foundation Models**: EFM3D/EVL (frozen backbone for 3D feature extraction)
- **Datasets**: Aria Synthetic Environments (ASE) - 100K synthetic indoor scenes with GT meshes
- **Project Specific Stack**: `atek` (Aria Training & Evaluation Toolkit), `efm3d` (EFM3D model implementation with various utilities), `projectaria_tools` (ASE dataset utilities)
- **3D Processing**: `pytorch3d`, `trimesh`, `pytransform3d`, `open3d`
- **Visualization**: `plotly`, `streamlit`, `matplotlib`
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Python Environment**: Always ensure to work in the conda environment `aria-nbv` located at `/home/jandu/miniforge3/envs/aria-nbv/bin/python`, Python 3.11. All libraries are installed as packages.

### Agentic Behaviors

- **On Initialization**:
  - **Always** run `make context` (in active `aria-nbv` conda env) to generate an up-to-date class diagram of the `oracle_rri` package and get an overview of all defined symbols for reference.
  - **Always** run `make context-dir-tree` right after `make context` to get the current oracle_rri directory tree snapshot.
  - **Always** read: `index.qmd`, `todos.qmd`.
  - **Optional** (depending on your task) read: `ase_dataset.qmd`, `resources.qmd`, `oracle_rri_impl.qmd`, `efm3d_implementation.qmd`, `efm3d_symbol_index.qmd`, `prj_aria_tools_impl.qmd`, `rri_computation.qmd` to get a comprehensive understanding of the project goals, architecture, and technical details.
- **Always** start with condensing the problem description, then do initial exploration of all potentially relvant files before presenting a rough outline of the solution together with termination and acceptance criteria then implement incrementally while testing along the way. Always maintain a list of tasks.
- **Always** follow code conventions (type hints, Google docstrings, Column enums)
- **Prefer** using existing external implementations rather than reinventing the wheel - when confronted with a new requirement or problem, We can always add new libraries! Feel free to test code ideas in the terminal before proceeding.
- **Follow** established patterns and aim for clear separation of concerns (modularity, single responsibility principle, config-as-factory pattern).
- **Ensure** code quality:
  - **For package code:** Run `ruff format <file>` -> `ruff check <file>` -> `pytest <file>` to ensure code style compliance after finishing work on a file; Work test-driven and run pytest for changed code.
- **Use** your **MCP tools** (upstash_context7 [get-library-docs, resolve-library-id], code-index [find_files, get_file_summary, get_file_watcher_status], github_mcp_server) to retrieve helpful context from external libraries or our own codebase when needed. `make context-dir-tree` can be used to get an overview of the project directory structure. And `make context-external` to get a summary of the `efm3d` and `atek` packages.
- **Use** your web search capabilities to find further sources (i.e. papers on arXiv, Wikipedia)
- **Regularly** step back and think about your alignment, high-level design and implementation strategy before diving into code changes. Checking your alignment with the project goals must be done by #think-ing about the problem at hand.
- **Frequenty** summarize your findings and understandings of the project and use your scratchpad in @.github/codex_memory.md. Keep track of your todos and progress.
- **Always** inspect all referenced symbols or files and get an initial understanding, then #think to plan your next steps before potentially gathering additional context or making code changes.
- **When in doubt**, ask for clarification or additional context rather than making assumptions or gether relevant context autonomously.
- **Never** terminate without confirming that all of your changes have been tested with real-life pytest scenearios - i.e. using real data, not just mocks: the full functionality of all modules must be verified in integration tests.
- **Never** make any changes that have not been requested or are outside the scope of the current task.
- **Always** keep the documentation up to date with any code changes!
- **Before terminating**: replace any temporary citation placeholders of the form `cite…` with either (a) direct markdown links to authoritative sources, or (b) valid bib references already present in `references.bib`. Do not leave the placeholder markers in committed text.
- **Finally**, before terminating, summarize all important findings, changes made, and any open suggestions that remain in your final report. All crucial finfings and suggestions must be documented in `.codex/<label for tasks>.md` for future reference. Here include potential problems in the current implementation, suggestions for future improvements, and any other relevant insights gained during your work that could benefit future contributors.

## Key Documentation Files

**Always reference these files** for project context and technical details:

- **[docs/index.qmd](../docs/index.qmd)**: Main project overview, vision, goals, and navigation hub
- **[docs/contents/todos.qmd](../docs/contents/todos.qmd)**: Current action items, priorities, and open questions
- **[docs/contents/roadmap.qmd](../docs/contents/roadmap.qmd)**: Long-term planning, milestones, and release sequencing
- **[docs/contents/resources.qmd](../docs/contents/resources.qmd)**: Central index of external links, API references, data sheets, and toolchains
- **[docs/contents/questions.qmd](../docs/contents/questions.qmd)**: Open research/design questions with historical answers and rationale
- **[docs/contents/ase_dataset.qmd](../docs/contents/ase_dataset.qmd)**: ASE dataset structure, ATEK data keys, download strategy, and mesh pairing rules
- **[docs/contents/setup.qmd](../docs/contents/setup.qmd)**: Environment bootstrap instructions (conda, system packages, dataset sync)
- **[docs/contents/literature/index.qmd](../docs/contents/literature/index.qmd)** plus topic pages (`efm3d.qmd`, `gen_nbv.qmd`, `vin_nbv.qmd`, `scene_script.qmd`): summaries of every paper we rely on
- **[docs/contents/theory/nbv_background.qmd](../docs/contents/theory/nbv_background.qmd)** and **[docs/contents/theory/rri_theory.qmd](../docs/contents/theory/rri_theory.qmd)**: mathematical derivations for NBV, Chamfer/RRI, and RRI normalization
- **[docs/contents/theory/semi-dense-pc.qmd](../docs/contents/theory/semi-dense-pc.qmd)** and **[docs/contents/theory/surface_metrics.qmd](../docs/contents/theory/surface_metrics.qmd)**: SLAM point properties & mesh metrics
- **Implementation series**:
  - **[docs/contents/impl/overview.qmd](../docs/contents/impl/overview.qmd)**: dependency graph across oracle_rri/geometry/viz modules
  - **[docs/contents/impl/oracle_rri_impl.qmd](../docs/contents/impl/oracle_rri_impl.qmd)**: Complete Oracle RRI implementation guide with function signatures and usage
  - **[docs/contents/impl/oracle_rri_class.qmd](../docs/contents/impl/oracle_rri_class.qmd)**: Planned package layout and config flow for the oracle module
  - **[docs/contents/impl/atek_implementation.qmd](../docs/contents/impl/atek_implementation.qmd)**: ATEK-specific utilities, key maps, and downloader interfaces
  - **[docs/contents/impl/efm3d_implementation.qmd](../docs/contents/impl/efm3d_implementation.qmd)** and **[docs/contents/impl/efm3d_symbol_index.qmd](../docs/contents/impl/efm3d_symbol_index.qmd)**: EFM3D utilities reference & exhaustive symbol catalog
  - **[docs/contents/impl/prj_aria_tools_impl.qmd](../docs/contents/impl/prj_aria_tools_impl.qmd)**: Project Aria tooling (trajectory, calibration, GT mesh access)
  - **[docs/contents/impl/rri_computation.qmd](../docs/contents/impl/rri_computation.qmd)**: Metrics, Chamfer distance recipes, and numerical stability notes
  - **[docs/contents/impl/efm3d_symbol_index.qmd](../docs/contents/impl/efm3d_symbol_index.qmd)**: (repeat) central reference for constants/classes—use it whenever touching EFM3D
- **Experiments**: **[docs/contents/experiments/findings.qmd](../docs/contents/experiments/findings.qmd)** tracks every evaluation run and key takeaways
- **Glossary**: **[docs/contents/glossary.qmd](../docs/contents/glossary.qmd)** defines project-specific terminology (RRI, OBB, candidate pose taxonomies)
- **[docs/contents/impl/atek_implementation.qmd](../docs/contents/impl/atek_implementation.qmd)** + **[docs/contents/impl/prj_aria_tools_impl.qmd](../docs/contents/impl/prj_aria_tools_impl.qmd)** are mandatory when modifying dataset loaders or downloaders
- **[docs/contents/questions.qmd](../docs/contents/questions.qmd)** + **[docs/contents/resources.qmd](../docs/contents/resources.qmd)** for context when answering design questions
- **[notebooks/ase_oracle_rri_simplified.ipynb](../notebooks/ase_oracle_rri_simplified.ipynb)**: Working Oracle RRI implementation with all fixes applied

## Style Guide

**General Guidelines**:
- ✓ Config classes inherit from our (e.g., `BaseConfig`)
- ✓ All functional classes (targets) and models instantiated via `my_config.setup_target()`
- ✓ Provide doc-strings for all relvant fields in pydantic classes or dataclasses, rather than using `Field(..., description="...")`. Don't use `Field(..., )` for primitive fields unless necessary (i.e, when `defaul_factory` is required). Example:
    ```py
    class MyConfig(BaseConfig):
        my_bool: bool = True
        """Whether to enable the awesome feature."""
    ```
- ✓ Prefer vectorized approaches over functional approaches over comprehensions over loops
- ✓ Use `pathlib.Path` for all filesystem paths
- ✓ Work test-driven; every new feature must have corresponding tests in `tests` using `pytest`
- ✓ Prefer `match-case` over `if-elif-else` for multi-branch logic when applicable
- ✓ Prefer `Enum` for categorical variables over string literals
- ✓ Follow EFM3D/ATEK coordinate conventions (see `efm3d_symbol_index.qmd`)
- ✓ Use ARIA constants from `efm3d.aria.aria_constants` for dataset keys
- ✓ All poses must use `PoseTW` from `efm3d.aria.pose`
- ✓ All cameras must use `CameraTW` from `efm3d.aria.camera`
- ✓ Document tensor shapes and coordinate frames in comments
- ✓ Use `Console` from `oracle_rri.utils` for structured logging
- ✓ Perfer usage of existing utilities from `efm3d`, `atek`, and `projectaria_tools` over reimplementation

- ✓ **Typing**
    - All signatures must be typed; Use modern builtins (`list[str]`, `dict[str, Any]`)
    - Use `TYPE_CHECKING` guards for imports of types only used in annotations
    - Use `Literal` for constrained string values

- ✓ **Docstrings**: All public methods must have Google-style docstrings including type and shape for tensor/array arguments and return values


## Documentation and Paper Conventions

- `typst` for presentations and publications
- `quarto` for documentation site
- **Always** use `docs/references.bib` for bibliography management.
- **Always** use `make context-qmd-tree` when working on documentation to get an up-to-date overview of the documentation structure.
- **In documentation**, include links to relevant documentations pages or external references like Wikipedia when mentioning key concepts or algorithms for the first time. Cite them as `[Wikipedia :: Concept Name](https://en.wikipedia.org/wiki/...)` in the text and add the corresponding entry in `references.bib`.
- Aim for clarity and conciseness in explanations. Use diagrams or code snippets where appropriate to illustrate complex ideas.
- Aim for a high educational value in explanations, assuming the reader has a graduate-level understanding of computer vision and machine learning but may not be familiar with the specific techniques used in this project.
- **Always** ensure that the quarto project can be compiled wihtout issues, when making larger changes to the documentation run `quarto check` and `quarto render` to verify.

**Example (Typing + Docstring)**:
```python
from torch import Tensor

def compute_rri(
    P_t: Tensor,
    P_q: Tensor,
    gt_mesh_vertices: Tensor,
    gt_mesh_faces: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute Relative Reconstruction Improvement for candidate view.

    Args:
        P_t (Tensor["N 3", float32]): Current reconstruction point cloud (N points, XYZ).
        P_q (Tensor["M 3", float32]): Candidate view point cloud (M points, XYZ).
        gt_mesh_vertices (Tensor["V 3", float32]): Ground truth mesh vertices (V vertices, XYZ).
        gt_mesh_faces (Tensor["F 3", int64]): Ground truth mesh face indices (F faces, 3 indices).

    Returns:
        Tuple[Tensor, Tensor] containing:
            - Tensor['B num_classes H W', float32]: Output tensor after processing.
            - Tensor['B', float32]: Auxiliary output tensor.
    """
    ...
```

### Architecture & Design Patterns

- **Config-as-Factory:** Every runtime object is created via their Pydantic config's `setup_target()` method. Never instantiate runtime classes directly. Use `field_validator` and `model_validator` decorators for cleanly structured validation within configs.
-
    - Examples:

    ```python
    from pydantic import BaseModel, Field
    from oracle_rri.lit_utils import BaseConfig

    class MyComponentConfig(BaseConfig["MyComponent"]):
        target: type["MyComponent"] = Field(default_factory=lambda: MyComponent, exclude=True)
        """Factory target for the config. This should be the runtime class that
        will be instantiated by `setup_target()` and is excluded from serialization."""

        # Config fields
        learning_rate: float = 1e-3
        """Learning rate for optimizer."""
        batch_size: int = 32
        """Mini-batch size used for training and evaluation loops."""

    class MyComponent:
        def __init__(self, config: MyComponentConfig):
            self.config = config
    ```

    - Config Composition:

    ```python
    class ExperimentConfig(BaseConfig):
        trainer_config: TrainerFactoryConfig
        """Configuration for the trainer factory (optimizer, scheduler, devices)."""
        module_config: DocClassifierConfig
        """Configuration for the model module (architecture, heads, loss weights)."""
        datamodule_config: DocDataModuleConfig
        """Configuration for data ingestion (datasets, transforms, batch sizing)."""

        def setup_target(self) -> tuple[Trainer, LightningModule, LightningDataModule]:
            trainer = self.trainer_config.setup_target()
            module = self.module_config.setup_target()
            datamodule = self.datamodule_config.setup_target()
            return trainer, module, datamodule
    ```

- **Data Views**

    ```python
    @dataclass(slots=True)
    class EfmCameraView:
        """Camera stream in EFM schema.

        Attributes:
            images: ``Tensor["F C H W", float32]``
                - RGB or mono images normalised to ``[0, 1]``.
                - ``F`` frames at the fixed snippet length (usually 20 for 2 s @10 Hz).
            calib: :class:`CameraTW`
                - Batched intrinsics/extrinsics (fisheye624 params, valid radius, exposure, gain).
                - ``CameraTW.tensor`` shape ``(F, 34)`` storing projection params and pose.
            ...
        """

        images: Tensor
        """``Tensor["F C H W", float32]`` normalized camera images in Aria RDF frame."""
        calib: CameraTW
        """Per-frame camera intrinsics/extrinsics (`CameraTW.tensor` shape ``(F,34)``)."""
        ...
    ```

- **Console Logging:** Use `Console` from `oracle_rri.utils` for structured, context-aware logging:

    ```python
    from oracle_rri.utils import Console

    console = Console.with_prefix(self.__class__.__name__, "setup_target")
    console.set_verbose(self.verbose).set_debug(self.is_debug)

    console.log("Starting setup...")          # Info when verbose=True
    console.warn("Deprecated parameter")       # Warning + caller stack
    console.error("Invalid configuration")     # Error + caller stack
    console.dbg("Internal state: ...")         # Debug when is_debug=True
    console.plog(complex_obj)                  # Pretty-print with devtools
    ```

## Further Technical Details

### AseEfmDataset usage

```python
from oracle_rri.data import AseEfmDatasetConfig

cfg = AseEfmDatasetConfig() # all config options have valid defaults!
cfg.inspect() # to get an over overvierw of all config options
dataset = cfg.setup_target()

sample = next(iter(dataset)) # of type oracle_rri.data.EfmSnippetView
sample.efm              # raw EFM3D snippet dict (zero-copy)
sample.mesh             # trimesh.Trimesh or None; sample.has_mesh for boolean
sample.camera_rgb       # EfmCameraView (images, calib, time_ns, frame_ids)
sample.camera_slam_left # EfmCameraView for SLAM-L stream
sample.camera_slam_right# EfmCameraView for SLAM-R stream
sample.trajectory       # EfmTrajectoryView (t_world_rig, time_ns, gravity_in_world, final_pose)
sample.semidense        # EfmPointsView (points_world, dist_std, inv_dist_std, volume_min/max, lengths)
sample.obbs             # EfmObbView or None (padded ObbTW + hz)
sample.gt               # EfmGTView (timestamps -> cameras -> OBB GT fields)

# device move without cloning when already on device
sample = sample.to("cuda")  # or .to("cpu")

# iterate GT timestamps and per-camera boxes
for ts in sample.gt.timestamps:
    cams = sample.gt.cameras_at(ts)
    rgb_obb = cams["camera-rgb"]
    print(ts, rgb_obb.object_dimensions.shape)
```


### Coordinate System Conventions

**Critical**: Follow EFM3D/ATEK coordinate system conventions strictly:

- **World Frame**: Global fixed coordinate system (GT meshes, SLAM points)
- **Rig/Device Frame**: Moves with the AR headset
- **Camera Frame**: Individual camera sensor (fixed relative to rig)
- **Voxel Frame**: Volumetric grid coordinate system

**Transformation Notation**:
- `T_A_B`: Transform from frame B to frame A
- `t_device_camera`: Transform from camera to device/rig
- `ts_world_device`: Time series of world-to-device transforms
- Always use `PoseTW` for SE(3) poses, never raw matrices

### ATEK Data Format

**Key Naming**: `<prefix>#<identifier>+<parameter>`

**Prefixes**:
- `mtd`: **M**otion **T**rajectory **D**ata (device poses)
- `mfcd`: **M**ulti-**F**rame **C**amera **D**ata (camera streams)
- `msdpd`: **M**ulti-**S**emi-**D**ense **P**oint **D**ata (SLAM points)

### EFM Snippet View Quick Reference

The EFM-facing dataset yields `EfmSnippetView` with these zero-copy properties:
- `camera_rgb`, `camera_slam_left`, `camera_slam_right` → `EfmCameraView` (`images`, `calib`, `time_ns`, `frame_ids`)
- `trajectory` → `EfmTrajectoryView` (`t_world_rig`, `time_ns`, `gravity_in_world`, `final_pose`)
- `semidense` → `EfmPointsView` (`points_world`, `dist_std`, `inv_dist_std`, `time_ns`, `volume_min`, `volume_max`, `lengths`)
- `obbs` → `EfmObbView` (padded `ObbTW` + `hz`) or `None`
- `gt` → `EfmGTView` with per-timestamp `EfmGtTimestampView` and per-camera `EfmGtCameraObbView` (`category_ids/names`, `instance_ids`, `object_dimensions`, `ts_world_object`)
- `mesh` / `has_mesh` → optional GT mesh attached to the scene

All fields are views over the dict produced by `load_atek_wds_dataset_as_efm`; call `.to(...)` on the snippet or its sub-views to move tensors without cloning when possible.


## Context7 Library Documentation

- `/facebookresearch/atek` - Aria Training and Evaluation Kit
- `/websites/facebookresearch_github_io_projectaria_tools` - Project Aria Tools docs
- `/facebookresearch/efm3d` - Egocentric Foundation Models for 3D understanding
- `/mikedh/trimesh` - Mesh processing and analysis
- `/rocm/pytorch` - PyTorch deep learning framework
- `/facebookresearch/pytorch3d` - 3D deep learning operations
- `/plotly/plotly.py` - Interactive 3D visualizations
- `/dfki-ric/pytransform3d` - 3D transformations and coordinate frames
- `/isl-org/open3d` - 3D data processing library
- `/pydantic/pydantic` - Data validation and settings management
- `/websites/streamlit_io` - Web app framework for data apps
- `/websites/typst_app` - Presentations and publications
- `/websites/quarto` - For our documentation site

**Note**: For EFM3D, ATEK, and ProjectAria tools, refer to the vendored source code in `external/` and the symbol index at `docs/contents/impl/efm3d_symbol_index.qmd`.

---

## Warning

**This document may sometimes be out of date. Hence, references to file, classes, functions or design patterns may not always be accurate. Always verify against the actual codebase and project documentation - here `make context*` provide up-to-date snapshots of the code structure and symbol definitions that can be considered ground truth.**

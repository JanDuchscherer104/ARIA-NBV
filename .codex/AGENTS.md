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
- **Python Environment**: Use the uv-managed virtualenv at `oracle_rri/.venv` (Python 3.11). If it is missing or stale, recreate it with
  `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d`.

### Agentic Behaviors

## 0. Quickstart (Recommended)

### Commands (use the uv-managed venv)

- Context: `make context` then `make context-dir-tree`
- Tests (avoid system python): `uv run pytest <path>` or `oracle_rri/.venv/bin/python -m pytest <path>`
- Lint/format: `ruff format <file>` then `ruff check <file>`

### Common gotchas (hit before)

- **Validation is disabled by default**: set `trainer_config.enable_validation=true` (otherwise Lightning forces `limit_val_batches=0` and `check_val_every_n_epoch=0`).
- **Pydantic defaults**: never set `Field(default=<callable>)` unless you intend to store the callable; use `Field(default_factory=...)` for computed defaults.
- **Interpreter mismatch**: running `pytest` with the system python can miss deps (e.g. `power_spherical`); prefer `uv run` / `oracle_rri/.venv`.
- **Offline cache splits are file-backed**: `train_index.jsonl` / `val_index.jsonl` may be created/updated when loading caches with `split="train"/"val"`.

### Offline cache indexing / splitting

- Rebuild indices from `samples/*.pt` (also rebuilds randomized split): call `rebuild_cache_index(cache_dir=..., train_val_split=..., rng_seed=...)` from `oracle_rri.data.offline_cache`.
- Training/validation from cache uses `OracleRriCacheDatasetConfig.train_val_split` + `split="train"/"val"`; `VinDataModule` will auto-derive `val_cache` when split is active.

- **On Initialization**:
  - **Always** run `make context` (it uses the uv `.venv` automatically and prints which interpreter is used; if `.venv` is missing, run the sync command above first) to generate an up-to-date class diagram of the `oracle_rri` package. *Note*, an offline snapshot is stored in `.codex/codex_make_context.md`.
  - **Always** run `make context-dir-tree` right after `make context` to get the current oracle_rri directory tree snapshot.
  - **Always** read: `index.qmd`, `todos.qmd`
  - **Always** start with defining and clarifying the task at hand, deciding wether to compress your context and wether to gather additional context from relevant files or external libraries (git-library-docs MCP tool) before proceeding.
  - The user prompt may contain *Termination Criteria* that must be strictly followed by defining a verifiable condition for successful task completion - this often involves writing and executing tests to ensure that *all* potential issues or or failure modes have been addressed and you can confidently declare the initial request as done. It is often a good idea to test the assess interpretable (intermediate) results in quick cli experiments when debugging, this allows for incremental verification of your changes.i
  - **Never** too definsively - i.e. assume a properly working environment with all dependencies installed unless explicitly told otherwise.
  - **Optional** (depending on your task) read: `ase_dataset.qmd`, `resources.qmd`, `oracle_rri_impl.qmd`, `efm3d_implementation.qmd`, `efm3d_symbol_index.qmd`, `prj_aria_tools_impl.qmd`, `rri_computation.qmd` to get a comprehensive understanding of the project goals, architecture, and technical details.
- **Always** start with condensing the problem description, then do initial exploration of all potentially relvant files before presenting a rough outline of the solution together with termination and acceptance criteria then implement incrementally while testing along the way. Always maintain a list of tasks.
- **NEVER** run "git restore" on any file. **Always** asume that other agents or humans may have made changes to the codebase that you are not aware of. Don't concern yourself with changes outside your current task scope!
- **Always** follow code conventions (type hints, Google docstrings, Column enums)
- **Prefer** using existing external implementations rather than reinventing the wheel - when confronted with a new requirement or problem, We can always add new libraries! Feel free to test code ideas in the terminal before proceeding.
- **Follow** established patterns and aim for clear separation of concerns (modularity, single responsibility principle, config-as-factory pattern).
- **Ensure** code quality:
  - **For package code:** Run `ruff format <file>` -> `ruff check <file>` -> `pytest <file>` to ensure code style compliance after finishing work on a file; Work test-driven and run pytest for changed code.
- **Use** your **MCP tools** (upstash_context7 [get-library-docs, resolve-library-id], code-index [find_files, get_file_summary, get_file_watcher_status], github_mcp_server) to retrieve helpful context from external libraries or our own codebase when needed. `make context-dir-tree` can be used to get an overview of the project directory structure. And `make context-external` to get a summary of the `efm3d` and `atek` packages.
- **Use** your web search capabilities to find further sources (i.e. papers on arXiv, Wikipedia)
- **Regularly** step back and think about your alignment, high-level design and implementation strategy before diving into code changes. Checking your alignment with the project goals must be done by #think-ing about the problem at hand.
- **Always** inspect all referenced symbols or files and get an initial understanding, then #think to plan your next steps before potentially gathering additional context or making code changes.
- **When in doubt**, ask for clarification or additional context rather than making assumptions or gether relevant context autonomously.
- **Never** terminate without confirming that all of your changes have been tested with real-life pytest scenearios - i.e. using real data, not just mocks: the full functionality of all modules must be verified in integration tests.
- **Never** make any changes that have not been requested or are outside the scope of the current task.
- **Always** keep the documentation up to date with any code changes!
- **When editing diagrams**: validate Mermaid/Quarto locally with `quarto render docs/contents/impl/aria_nbv_overview.qmd --to html` (no absolute path in `--output`). Use `{mermaid}` fences, `<br/>` for line breaks inside labels, and avoid math-style braces/subscripts in node IDs to keep Mermaid 11+ happy.
- **Before terminating**: replace any temporary citation placeholders of the form `cite…` with either (a) direct markdown links to authoritative sources, or (b) valid bib references already present in `references.bib`. Do not leave the placeholder markers in committed text.
- **Always** add relevant references (optimally arxiv/bibtex entires) to `docs/references.bib` when mentioning key concepts, algorithms, or prior work in documentation or publications.
- **Finally**, before terminating, summarize all important findings, changes made, and any open suggestions that remain in your final report. All crucial finfings and suggestions must be documented in `.codex/<label for tasks>.md` for future reference. Here include potential problems in the current implementation, suggestions for future improvements, and any other relevant insights gained during your work that could benefit future contributors.

### Context7 library IDs (for diagrams/docs)

Keep resolved IDs handy for `{mermaid}` diagrams or documentation references:

```{mermaid}
flowchart LR
    ctx[Context7]
    ctx --> m["/mermaid-js/mermaid"]
    ctx --> q["/websites/quarto"]
```

### Mermaid validation workflow

- For any diagram edits, first validate standalone with mermaid-cli:
  `npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg`
  (labels that include spaces/parentheses should be wrapped in quotes inside edge labels; avoid braces/subscripts in IDs; use `<br/>` for line breaks).
- After it renders, copy the verified content into the `{mermaid}` block in the relevant `.qmd`, then run `quarto render <file>.qmd --to html` as a final check.

## Key Documentation Files

**Always reference these files** for project context and technical details:

- **docs/index.qmd**: Main project overview, vision, goals, and navigation hub.
- **docs/contents/todos.qmd** and **docs/contents/roadmap.qmd**: Current action items and milestones.
- **docs/contents/questions.qmd**: Open research/design questions with rationale.
- **docs/contents/resources.qmd** with **docs/contents/setup.qmd**: External links plus environment/bootstrap instructions.
- **docs/contents/ase_dataset.qmd**: ASE modality layout, ATEK key mapping, download/mesh pairing rules.
- **Theory set** — **docs/contents/theory/nbv_background.qmd**, **rri_theory.qmd**, **surface_metrics.qmd**, **semi-dense-pc.qmd**: NBV/RRI derivations, Chamfer metrics, SLAM point properties.
- **Literature hub** — **docs/contents/literature/index.qmd** and topic pages (`efm3d.qmd`, `vin_nbv.qmd`, `gen_nbv.qmd`, `scene_script.qmd`): Paper summaries and pointers.
- **Internal implementation guides** — **docs/contents/impl/overview.qmd**, **data_pipeline_overview.qmd**, **class_diag.qmd**, **aria_nbv_package.qmd**, **oracle_rri_impl.qmd**, **rri_computation.qmd**: Package layout, pipelines, UML, and oracle/metric details.
- **External stack references** — **docs/contents/ext-impl/efm3d_implementation.qmd**, **efm3d_symbol_index.qmd**, **atek_implementation.qmd**, **prj_aria_tools_impl.qmd**: Vendor code notes, symbol catalogs, and tooling specifics.
- **docs/contents/experiments/findings.qmd**: Log of evaluation runs and key takeaways.
- **docs/contents/glossary.qmd**: Project terminology (RRI, OBB, pose taxonomies).
- **notebooks/ase_oracle_rri_simplified.ipynb**: Reference notebook with the working oracle RRI pipeline.

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

- ✓ **Docstrings**: All public methods must have Google-style docst

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

Library IDs to use with `get-library-docs` MCP tool for external dependencies:

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
- `/websites/astral_sh_uv` - UV package manager

**Note**: For EFM3D, ATEK, and ProjectAria tools, refer to the vendored source code in `external/` and the symbol index at `docs/contents/impl/efm3d_symbol_index.qmd`.

---

## Warning

**This document may sometimes be out of date. Hence, references to file, classes, functions or design patterns may not always be accurate. Always verify against the actual codebase and project documentation - here `make context*` provide up-to-date snapshots of the code structure and symbol definitions that can be considered ground truth.**
rings including type and shape for tensor/array arguments and return values


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

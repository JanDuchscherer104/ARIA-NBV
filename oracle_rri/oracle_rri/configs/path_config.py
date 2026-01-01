from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator

from ..utils import Console, SingletonConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_root() -> Path:
    return PROJECT_ROOT


class PathConfig(SingletonConfig):
    """Centralise all filesystem locations for the oracle_rri project."""

    root: Path = Field(
        default_factory=_default_root,
    )
    "Project root."
    data_root: Path = Field(default_factory=lambda: Path(".data"))
    """Root directory for all data (downloaded datasets, meshes, etc.)."""

    data_root_massive: Path | None = Field(
        default_factory=lambda: Path("/mnt/e/wsl-data/ase-atek-nbv"),
    )
    """Optional root directory for storing further large datasets, caches or artifacts."""

    checkpoints: Path = Field(default_factory=lambda: Path(".logs") / "checkpoints")
    """Directory used by Lightning checkpoints."""

    wandb: Path = Field(default_factory=lambda: Path(".logs") / "wandb")
    """Directory used by Weights & Biases for local run artifacts."""

    optuna: Path = Field(default_factory=lambda: Path(".logs") / "optuna")
    """Directory used by Optuna for local study storage (SQLite)."""

    configs_dir: Path = Field(default_factory=lambda: Path(".configs"))
    """Directory containing exported experiment/configuration files (TOML, etc.)."""

    # ASE-specific paths
    url_dir: Path = Field(default_factory=lambda: Path(".data") / "aria_download_urls")
    """Directory containing ASE download URL JSON files."""

    metadata_cache: Path = Field(
        default_factory=lambda: Path(".data") / "ase_metadata.json",
    )
    """Path to cached ASE metadata JSON."""

    offline_cache_dir: Path = Field(default_factory=lambda: Path("offline_cache"))

    ase_meshes: Path = Field(default_factory=lambda: Path(".data") / "ase_meshes")
    """Directory for downloaded ASE ground truth meshes."""

    processed_meshes: Path = Field(
        default_factory=lambda: Path(".data") / "ase_meshes_processed",
    )
    """Directory for cropped/simplified meshes persisted for reuse."""

    external_dir: Path = Field(default=Path("external"))

    @classmethod
    def _resolve_path(cls, value: str | Path, info: ValidationInfo) -> Path:
        root = info.data.get("root", PROJECT_ROOT)
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.expanduser().resolve()

    @classmethod
    def _ensure_dir(cls, path: Path, field_name: str | None) -> Path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            Console.with_prefix(cls.__name__, field_name or "").log(
                f"Created directory: {path}",
            )
        return path

    @field_validator("root", mode="before")
    @classmethod
    def _validate_root(cls, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Configured project root '{path}' does not exist.")
        return path

    @field_validator(
        "checkpoints",
        "wandb",
        "optuna",
        "data_root",
        "configs_dir",
        "url_dir",
        "ase_meshes",
        "processed_meshes",
        "external_dir",
        "data_root_massive",
        mode="before",
    )
    @classmethod
    def _resolve_dirs(
        cls,
        value: str | Path | None,
        info: ValidationInfo,
    ) -> Path | None:
        if value is None:
            return None
        path = cls._resolve_path(value, info)
        if info.field_name == "url_dir":
            return path  # do not auto-create; callers may want missing detection
        return cls._ensure_dir(path, info.field_name)

    @field_validator("metadata_cache", mode="before")
    @classmethod
    def _resolve_metadata_cache(cls, value: str | Path, info: ValidationInfo) -> Path:
        """Resolve metadata cache path but don't create it yet."""
        return cls._resolve_path(value, info)

    @field_validator("offline_cache_dir", mode="before")
    @classmethod
    def _resolve_rel_massive_dir(
        cls,
        value: str | Path | None,
        info: ValidationInfo,
    ) -> Path | None:
        if value is None:
            return None
        path = cls._resolve_path(
            Path(info.data.get("data_root_massive", "")) / value,
            info,
        )
        return path

    def resolve_checkpoint_path(self, path: str | Path | None) -> Path | None:
        """Resolve a checkpoint path relative to the checkpoints directory.

        Args:
            path: Checkpoint path (absolute, relative, or None).

        Returns:
            Resolved absolute path, or None if input is None/empty.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        if path in (None, ""):
            return None

        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoints / checkpoint_path

        checkpoint_path = checkpoint_path.expanduser().resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' does not exist.",
            )

        if not checkpoint_path.suffix == ".ckpt":
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' is not a .ckpt file.",
            )

        return checkpoint_path

    def resolve_mesh_path(self, scene_id: str) -> Path:
        """Resolve path to GT mesh for a scene.

        Args:
            scene_id: Scene identifier (e.g., "82832")

        Returns:
            Path to mesh file (may not exist yet)
        """
        return self.ase_meshes / f"scene_ply_{scene_id}.ply"

    def resolve_processed_mesh_path(
        self,
        scene_id: str,
        simplification_ratio: float,
        is_crop: bool,
        spec_hash: str,
    ) -> Path:
        """Resolve path for a processed (cropped/simplified) mesh artifact.

        Args:
            scene_id: ASE scene identifier.
            snippet_id: Optional snippet/shard identifier for disambiguation.
            spec_hash: Short hash of the processing specification.

        Returns:
            Destination path where the processed mesh should be stored.
        """
        filename = f"scene_{scene_id}_{simplification_ratio}_{'crop' if is_crop else 'nocrop'}_{spec_hash}.ply"
        return self.processed_meshes / filename

    def resolve_atek_data_dir(self, config_name: str = "efm") -> Path:
        """Resolve path to ATEK data directory for a config.

        Args:
            config_name: ATEK config name (e.g., "efm", "efm_eval")

        Returns:
            Path to ATEK data directory
        """
        atek_dir = self.data_root / f"ase_{config_name}"
        atek_dir.mkdir(parents=True, exist_ok=True)
        return atek_dir

    def get_atek_source_path(self) -> Path:
        """Get path to vendored ATEK source.

        Returns:
            Path to external/ATEK directory

        Raises:
            FileNotFoundError: If ATEK is not found
        """
        atek_path = self.root / self.external_dir / "ATEK"
        if not atek_path.exists():
            raise FileNotFoundError(
                f"ATEK source not found at {atek_path}. Please ensure external/ATEK is cloned.",
            )
        return atek_path

    def get_atek_url_json_path(
        self,
        json_filename: str = "AriaSyntheticEnvironment_ATEK_download_urls.json",
    ) -> Path:
        """Get path to ATEK download URLs JSON.

        Args:
            json_filename: Name of the ATEK URL JSON file.

        Returns:
            Path to the ATEK URL JSON file.
        """
        json_path = (self.url_dir / json_filename).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"ATEK URL JSON not found at {json_path}.")
        return json_path

    # --------------------------------------------------------------------- runtime path helpers
    def resolve_under_root(
        self,
        path: str | Path,
        *,
        base_dir: Path | None = None,
    ) -> Path:
        """Resolve a user-provided path relative to the project root.

        Args:
            path: Path to resolve (absolute or relative).
            base_dir: Optional base directory to resolve relative paths against.
                Defaults to :attr:`root`.

        Returns:
            Absolute, expanded, resolved path.
        """
        base = base_dir if base_dir is not None else self.root
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = base / resolved
        return resolved.expanduser().resolve()

    def resolve_run_dir(self, out_dir: str | Path) -> Path:
        """Resolve a run output directory and ensure it exists."""
        resolved = self.resolve_under_root(out_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def resolve_artifact_path(
        self,
        path: str | Path,
        *,
        expected_suffix: str | None = None,
        create_parent: bool = True,
    ) -> Path:
        """Resolve an artifact file path under the project root.

        Args:
            path: Path to resolve (absolute or relative).
            expected_suffix: Optional required suffix (e.g. ".json", ".pt", ".toml").
            create_parent: Create the parent directory when True.

        Returns:
            Absolute, expanded, resolved path.
        """
        resolved = self.resolve_under_root(path)
        if expected_suffix is not None and resolved.suffix != expected_suffix:
            raise ValueError(
                f"Expected path with suffix {expected_suffix!r}, got {resolved}.",
            )
        if create_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def resolve_config_toml_path(
        self,
        path: str | Path,
        *,
        must_exist: bool = True,
    ) -> Path:
        """Resolve a TOML experiment config path.

        Relative paths are resolved as:
        - bare filenames (no parent) are interpreted as under :attr:`configs_dir`
        - other relative paths are interpreted as under :attr:`root`
        """
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = (self.configs_dir / cfg_path) if cfg_path.parent == Path() else (self.root / cfg_path)
        cfg_path = cfg_path.expanduser().resolve()
        if cfg_path.suffix != ".toml":
            raise ValueError(f"Config path must be a .toml file, got {cfg_path}.")
        if must_exist and not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        return cfg_path

    def resolve_optuna_study_uri(self, study_name: str) -> str:
        """Resolve a SQLite Optuna storage URI for a given study name."""
        db_path = (self.optuna / f"{study_name}.db").expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

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

    checkpoints: Path = Field(default_factory=lambda: Path(".logs") / "checkpoints")
    """Directory used by Lightning checkpoints."""

    configs_dir: Path = Field(default_factory=lambda: Path(".configs"))
    """Directory containing exported experiment/configuration files (TOML, etc.)."""

    # ASE-specific paths
    url_dir: Path = Field(default_factory=lambda: Path(".data") / "aria_download_urls")
    """Directory containing ASE download URL JSON files."""

    metadata_cache: Path = Field(default_factory=lambda: Path(".data") / "ase_metadata.json")
    """Path to cached ASE metadata JSON."""

    ase_meshes: Path = Field(default_factory=lambda: Path(".data") / "ase_meshes")
    """Directory for downloaded ASE ground truth meshes."""

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
            Console.with_prefix(cls.__name__, field_name or "").log(f"Created directory: {path}")
        return path

    @field_validator("root", mode="before")
    @classmethod
    def _validate_root(cls, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Configured project root '{path}' does not exist.")
        return path

    @field_validator("checkpoints", "data_root", "configs_dir", "url_dir", "ase_meshes", "external_dir", mode="before")
    @classmethod
    def _resolve_dirs(cls, value: str | Path, info: ValidationInfo) -> Path:
        path = cls._resolve_path(value, info)
        return cls._ensure_dir(path, info.field_name)

    @field_validator("metadata_cache", mode="before")
    @classmethod
    def _resolve_metadata_cache(cls, value: str | Path, info: ValidationInfo) -> Path:
        """Resolve metadata cache path but don't create it yet."""
        return cls._resolve_path(value, info)

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
            raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")

        if not checkpoint_path.suffix == ".ckpt":
            raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' is not a .ckpt file.")

        return checkpoint_path

    def resolve_mesh_path(self, scene_id: str) -> Path:
        """Resolve path to GT mesh for a scene.

        Args:
            scene_id: Scene identifier (e.g., "82832")

        Returns:
            Path to mesh file (may not exist yet)
        """
        return self.ase_meshes / f"scene_ply_{scene_id}.ply"

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
            raise FileNotFoundError(f"ATEK source not found at {atek_path}. Please ensure external/ATEK is cloned.")
        return atek_path

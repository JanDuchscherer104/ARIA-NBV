"""Pydantic configuration models for the oracle RRI pipeline."""

from pathlib import Path

from pydantic import Field, field_validator

from .utils import BaseConfig

DEFAULT_DATA_ROOT = Path(".data")
DEFAULT_WDS_ROOT = DEFAULT_DATA_ROOT / "ase_atek"
DEFAULT_MESH_ROOT = DEFAULT_DATA_ROOT / "ase_meshes"
DEFAULT_OUTPUT_ROOT = Path("outputs/oracle_rri")
DEFAULT_ATEK_MANIFEST = DEFAULT_DATA_ROOT / "aria_download_urls/AriaSyntheticEnvironment_ATEK_download_urls.json"
DEFAULT_MESH_MANIFEST = DEFAULT_DATA_ROOT / "aria_download_urls/ase_mesh_download_urls.json"


class DatasetPaths(BaseConfig):
    """Filesystem layout for ASE/ATEK assets used during oracle-RRI computation."""

    atek_manifest: Path = Field(default=DEFAULT_ATEK_MANIFEST)
    mesh_manifest: Path = Field(default=DEFAULT_MESH_MANIFEST)
    wds_root: Path = Field(default=DEFAULT_WDS_ROOT)
    mesh_root: Path = Field(default=DEFAULT_MESH_ROOT)

    @field_validator("atek_manifest", "mesh_manifest")
    @classmethod
    def validate_manifest(cls, value: Path) -> Path:
        if not value.exists():
            raise FileNotFoundError(f"Manifest not found: {value}. Ensure downloads have been executed.")
        return value

    @field_validator("wds_root", "mesh_root")
    @classmethod
    def ensure_directory(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


class OracleConfig(BaseConfig):
    """Top-level configuration for shared oracle-RRI services."""

    dataset: DatasetPaths = Field(default_factory=DatasetPaths)
    output_root: Path = Field(default=DEFAULT_OUTPUT_ROOT)
    device: str = Field(default="cpu", description="Torch device used for heavy operations")
    batch_size: int = Field(default=1, ge=1)
    max_sequences: int | None = Field(default=None)
    max_snippets: int | None = Field(default=None)

    @field_validator("output_root")
    @classmethod
    def ensure_output_root(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("batch_size must be >= 1")
        return value

    @field_validator("max_sequences", "max_snippets")
    @classmethod
    def _check_positive(cls, value: int | None, info):
        if value is not None and value < 1:
            raise ValueError(f"{info.field_name} must be positive when provided")
        return value


__all__ = ["DatasetPaths", "OracleConfig"]

"""Download orchestration for ASE dataset meshes + ATEK snippets."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Literal

import requests
from atek.data_download.atek_data_store_download import download_atek_wds_sequences
from pydantic import Field, field_validator
from pydantic_settings import CLI_SUPPRESS, BaseSettings, CliSuppress, SettingsConfigDict
from tqdm import tqdm

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .metadata import ASEMetadata, SceneInfo


class ASEDownloaderConfig(BaseSettings, BaseConfig["ASEDownloader"]):
    """Configuration for ASE downloader with CLI support.

    Supports two CLI modes:
        1. Download mode (default): Download N scenes with meshes + ATEK snippets
        2. List mode: List available scenes

    Example (Programmatic):
        >>> config = ASEDownloaderConfig()
        >>> downloader = config.setup_target()
        >>> scenes = downloader.metadata.get_scenes(n=5, max_snippets=2)
        >>> downloader.download_scenes(scenes)

    Example (CLI - Download):
        $ python -m oracle_rri.data_handling.downloader --n_scenes=5 --max_snippets=2
        $ python -m oracle_rri.data_handling.downloader --n_scenes=10 --skip_meshes

    Example (CLI - List):
        $ python -m oracle_rri.data_handling.downloader list --n=10
    """

    # Internal fields (excluded from CLI)
    target: CliSuppress[type[ASEDownloader]] = Field(default_factory=lambda: ASEDownloader, description=CLI_SUPPRESS)
    mode: CliSuppress[Literal["download", "list"]] = Field(default="download", description=CLI_SUPPRESS)

    paths: PathConfig = Field(default_factory=PathConfig)
    # Core configuration
    verbose: bool = Field(default=True)
    """Enable verbose logging."""

    # JSON file configuration
    mesh_json_filename: str = Field(default="ase_mesh_download_urls.json")
    """Filename of the mesh download URLs JSON in url_dir."""

    atek_json_filename: str = Field(default="AriaSyntheticEnvironment_ATEK_download_urls.json")
    """Filename of the ATEK download URLs JSON in url_dir."""

    atek_config_name: Literal["efm", "efm_eval", "cubercnn", "cubercnn_eval"] = Field(default="efm")
    """ATEK configuration name."""

    # Download selection
    n_scenes: int = Field(default=5, ge=0)
    """Number of scenes to download (0 = all available scenes)."""

    max_snippets: int | None = Field(default=None, ge=1)
    """Maximum snippets per scene (None = all snippets)."""

    skip_meshes: bool = Field(default=False)
    """Skip downloading GT meshes (only download ATEK data)."""

    skip_atek: bool = Field(default=False)
    """Skip downloading ATEK WDS data (only download meshes)."""

    # ATEK downloader options (from atek_wds_data_downloader.py)
    train_val_split_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    """Train-validation split ratio for ATEK data."""

    random_seed: int = Field(default=42)
    """Random seed for shuffling ATEK data sequences."""

    download_wds_to_local: bool = Field(default=True)
    """Download WebDataset files to local storage (vs streaming URLs)."""

    overwrite: bool = Field(default=False)
    """Always overwrite existing ATEK files (re-download even if present)."""

    train_val_split_json_path: Path | None = Field(default=None)
    """Path to custom train/val split JSON (overrides train_val_split_ratio)."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="ASE_",
    )

    @field_validator("train_val_split_json_path", mode="before")
    @classmethod
    def _validate_split_json_path(cls, v: str | Path | None) -> Path | None:
        """Convert string to Path and validate existence."""
        if v is None:
            return None
        path = Path(v) if isinstance(v, str) else v
        if not path.exists():
            raise ValueError(f"Train/val split JSON not found: {path}")
        return path


class ASEDownloader:
    """Download ASE meshes + ATEK snippets."""

    def __init__(self, config: ASEDownloaderConfig):
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(config.verbose)

        # Parse metadata from JSONs
        self.metadata = ASEMetadata(config.paths.url_dir)

    def download_scenes(
        self,
        scenes: list[SceneInfo],
        download_meshes: bool = True,
        download_atek: bool = True,
    ) -> None:
        """Download meshes + ATEK data for scenes.

        Args:
            scenes: List of SceneInfo to download
            download_meshes: Download GT meshes
            download_atek: Download ATEK WDS snippets
        """
        if download_meshes:
            self._download_meshes(scenes)

        if download_atek:
            self._download_atek(scenes)

    def _download_meshes(self, scenes: list[SceneInfo]) -> None:
        """Download GT meshes from CDN.

        Adapted from ATEK's ase_mesh_downloader.py to align with our design patterns.
        Downloads, verifies SHA, extracts, and moves mesh files to output directory.
        """
        mesh_dir = self.config.paths.ase_meshes
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Check which meshes already exist
        scenes_to_download = []
        for scene in scenes:
            mesh_path = mesh_dir / f"scene_ply_{scene.scene_id}.ply"
            if mesh_path.exists():
                self.console.log(f"Scene {scene.scene_id}: mesh already exists, skipping")
            else:
                scenes_to_download.append(scene)

        if not scenes_to_download:
            self.console.log("All meshes already downloaded ✓")
            return

        self.console.log(
            f"Downloading {len(scenes_to_download)} GT meshes (skipping {len(scenes) - len(scenes_to_download)} existing)..."
        )

        # Download each mesh
        for scene in tqdm(scenes_to_download, desc="Downloading meshes", disable=not self.config.verbose):
            zip_filename = f"scene_ply_{scene.scene_id}.zip"
            ply_filename = f"scene_ply_{scene.scene_id}.ply"

            try:
                # Download ZIP from CDN
                response = requests.get(scene.mesh_url)
                response.raise_for_status()

                with open(zip_filename, "wb") as f:
                    f.write(response.content)

                # Verify SHA1 checksum
                with open(zip_filename, "rb") as f:
                    file_sha = hashlib.sha1(f.read()).hexdigest()

                if file_sha != scene.mesh_sha:
                    self.console.warn(
                        f"Scene {scene.scene_id}: SHA mismatch ({file_sha[:8]}... != {scene.mesh_sha[:8]}...)"
                    )
                    continue

                # Extract PLY from ZIP
                with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                    zip_ref.extractall()

                # Move to output directory
                if Path(ply_filename).exists():
                    Path(ply_filename).rename(mesh_dir / ply_filename)
                    self.console.log(f"Scene {scene.scene_id}: mesh downloaded ✓")
                else:
                    self.console.warn(f"Scene {scene.scene_id}: PLY not found in ZIP")

            except Exception as e:
                self.console.error(f"Scene {scene.scene_id}: download failed - {e}")

            finally:
                # Cleanup temporary files
                Path(zip_filename).unlink(missing_ok=True)
                Path(ply_filename).unlink(missing_ok=True)

        self.console.log("✓ Mesh downloads complete")

    def _download_atek(self, scenes: list[SceneInfo]) -> None:
        """Download ATEK WDS snippets using ATEK's download_atek_wds_sequences."""
        self.console.log(f"Downloading ATEK data for {len(scenes)} scenes...")

        # Load ATEK JSON and filter to selected scenes
        atek_json = self.config.paths.url_dir / self.config.atek_json_filename
        if not atek_json.exists():
            raise FileNotFoundError(f"ATEK JSON not found: {atek_json}")

        with atek_json.open() as f:
            atek_data = json.load(f)

        # Filter wds_file_urls to only include our scenes AND snippets
        config_data = atek_data["atek_data_for_all_configs"][self.config.atek_config_name]
        all_urls = config_data["wds_file_urls"]

        filtered_urls = {}
        for scene in scenes:
            if scene.scene_id not in all_urls:
                continue

            # Only include snippets that are in scene.snippet_ids
            scene_snippets = all_urls[scene.scene_id]
            filtered_snippets = {
                shard_key: shard_info
                for shard_key, shard_info in scene_snippets.items()
                if shard_key.replace("_tar", "") in scene.snippet_ids
            }

            if filtered_snippets:
                filtered_urls[scene.scene_id] = filtered_snippets

        if not filtered_urls:
            self.console.warn("No ATEK data found for selected scenes")
            return

        # Create filtered JSON for ATEK downloader
        filtered_data = {
            "raw_dataset_name": atek_data["raw_dataset_name"],
            "raw_dataset_release_version": atek_data["raw_dataset_release_version"],
            "atek_data_for_all_configs": {
                self.config.atek_config_name: {
                    **config_data,
                    "wds_file_urls": filtered_urls,
                }
            },
        }

        # Write temporary JSON
        tmp_json = self.config.paths.url_dir / f"_temp_atek_{self.config.atek_config_name}.json"
        with tmp_json.open("w") as f:
            json.dump(filtered_data, f)

        try:
            output_dir = self.config.paths.resolve_atek_data_dir(self.config.atek_config_name)

            # Call ATEK downloader directly (no subprocess)
            download_atek_wds_sequences(
                config_name=self.config.atek_config_name,
                input_json_path=str(tmp_json),
                train_val_split_ratio=self.config.train_val_split_ratio,
                random_seed=self.config.random_seed,
                output_folder_path=str(output_dir),
                max_num_sequences=len(filtered_urls) if filtered_urls else None,
                download_wds_to_local=self.config.download_wds_to_local,
                train_val_split_json_path=str(self.config.train_val_split_json_path)
                if self.config.train_val_split_json_path
                else None,
                overwrite=self.config.overwrite,
            )
            self.console.log("✓ ATEK data downloaded")

        finally:
            tmp_json.unlink(missing_ok=True)


# ============================================================================
# CLI Entry Points
# ============================================================================


def cli_download() -> None:
    """CLI entry point for downloading scenes.

    Usage:
        python -m oracle_rri.data_handling.downloader --n_scenes=5 --max_snippets=2
        python -m oracle_rri.data_handling.downloader --n_scenes=10 --skip_meshes
    """
    config = ASEDownloaderConfig()
    downloader = config.setup_target()

    # Get scenes
    n = None if config.n_scenes == 0 else config.n_scenes
    scenes = downloader.metadata.get_scenes(n=n, max_snippets=config.max_snippets)

    print(f"\n{'=' * 60}")
    print("ASE Dataset Downloader")
    print(f"{'=' * 60}")
    print(f"Downloading {len(scenes)} scenes:")
    for scene in scenes[:10]:  # Show first 10
        print(f"  - Scene {scene.scene_id}: {len(scene.snippet_ids)} snippets")
    if len(scenes) > 10:
        print(f"  ... and {len(scenes) - 10} more")

    total_snippets = sum(len(s.snippet_ids) for s in scenes)
    print(f"\nTotal: {len(scenes)} scenes, {total_snippets} snippets")

    # Download
    downloader.download_scenes(
        scenes=scenes,
        download_meshes=not config.skip_meshes,
        download_atek=not config.skip_atek,
    )

    print(f"\n{'=' * 60}")
    print("✓ Download complete!")
    print(f"{'=' * 60}\n")


def cli_list(n: int | None = None) -> None:
    """CLI entry point for listing scenes.

    Args:
        n: Number of scenes to list (None = all)

    Usage:
        python -m oracle_rri.data_handling.downloader list
        python -m oracle_rri.data_handling.downloader list --n=10
    """
    # Create config without CLI parsing for list mode
    config = ASEDownloaderConfig.model_validate({})
    downloader = config.setup_target()

    scenes = downloader.metadata.get_scenes(n=n)

    print(f"\n{'=' * 60}")
    print("ASE Dataset - Available Scenes")
    print(f"{'=' * 60}")
    print(f"Total scenes with GT meshes: {len(downloader.metadata.scenes)}")
    print(f"\nShowing {len(scenes)} scenes:")
    for scene in scenes:
        print(f"  Scene {scene.scene_id}: {len(scene.snippet_ids)} snippets")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Main CLI entry point with mode-based dispatching.

    Supports two modes:
        - 'download' (default): Download scenes with full CLI argument parsing
        - 'list': List available scenes with minimal arguments
    """
    # Check for 'list' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # Remove 'list' from argv
        sys.argv.pop(1)

        # Parse --n argument manually for list mode
        n = None
        if len(sys.argv) > 1:
            for i, arg in enumerate(sys.argv[1:], 1):
                if arg.startswith("--n="):
                    n = int(arg.split("=")[1])
                    sys.argv.pop(i)
                    break
                elif arg == "--n" and i + 1 < len(sys.argv):
                    n = int(sys.argv[i + 1])
                    sys.argv.pop(i)
                    sys.argv.pop(i)
                    break

        cli_list(n=n)
    else:
        # Download mode - use full pydantic-settings CLI parsing
        cli_download()


if __name__ == "__main__":
    main()

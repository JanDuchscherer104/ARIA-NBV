"""Lightweight CLI wrapper for ASE downloader used in tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from .downloader import ASEDownloaderConfig


class CLIDownloaderSettings:
    """Simple argparse-backed settings for CLI tests."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--url-dir", default=".data/aria_download_urls")
        parser.add_argument("--output-dir", default=".data")
        parser.add_argument("--verbose", default="true")
        parser.add_argument("--all-with-meshes", action="store_true")
        parser.add_argument("--scene-ids", nargs="*")
        parser.add_argument("--min-snippets", type=int, default=0)
        parser.add_argument("--config", default="efm")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--metadata-cache-path", default=None)
        parser.add_argument("--config-path", default=None)
        args = parser.parse_args()

        self.url_dir = Path(args.url_dir)
        self.output_dir = Path(args.output_dir)
        self.verbose = args.verbose.lower() != "false"
        self.all_with_meshes = args.all_with_meshes
        self.scene_ids = args.scene_ids
        self.min_snippets = args.min_snippets
        self.config = args.config
        self.overwrite = args.overwrite
        self.metadata_cache_path = args.metadata_cache_path
        self.config_path = args.config_path


def main() -> None:
    settings = CLIDownloaderSettings()
    cfg_kwargs = {"url_dir": settings.url_dir, "output_dir": settings.output_dir, "verbose": settings.verbose}
    config = ASEDownloaderConfig(**cfg_kwargs)
    downloader = config.setup_target()

    if settings.all_with_meshes:
        downloader.download_scenes_with_meshes(
            min_snippets=settings.min_snippets, config=settings.config, overwrite=settings.overwrite
        )
    else:
        # if scene_ids provided, filter metadata
        scenes = (
            [s for s in downloader.metadata.get_scenes() if s.scene_id in settings.scene_ids]
            if settings.scene_ids
            else downloader.metadata.get_scenes()
        )
        downloader.download_scenes(
            scenes=scenes,
            download_meshes=not config.skip_meshes,
            download_atek=not config.skip_atek,
        )


if __name__ == "__main__":
    main()

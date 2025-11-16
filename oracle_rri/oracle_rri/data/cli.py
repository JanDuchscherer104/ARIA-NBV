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
        self.meshes_only = getattr(args, "meshes_only", False)
        self.atek_only = getattr(args, "atek_only", False)


def main() -> None:
    settings = CLIDownloaderSettings()
    cfg_kwargs = {"url_dir": settings.url_dir, "output_dir": settings.output_dir, "verbose": settings.verbose}
    if settings.config_path:
        config = ASEDownloaderConfig.from_toml(settings.config_path)
    else:
        config = ASEDownloaderConfig(**cfg_kwargs)
    downloader = config.setup_target()

    def _flag(name: str) -> bool:
        val = getattr(settings, name, False)
        return bool(val) if isinstance(val, bool) else False

    meshes_only = _flag("meshes_only")
    atek_only = _flag("atek_only")

    if settings.all_with_meshes:
        downloader.download_scenes_with_meshes(
            min_snippets=settings.min_snippets, config=settings.config, overwrite=settings.overwrite
        )
    elif meshes_only:
        downloader.download_meshes(scene_ids=settings.scene_ids or [], overwrite=settings.overwrite)
    elif atek_only:
        downloader.download_atek(scene_ids=settings.scene_ids or [], overwrite=settings.overwrite)
    else:
        scenes = settings.scene_ids or []
        if not scenes:
            print("No download action specified; provide scene_ids or flags.")
            return
        downloader.download_scenes(
            scene_ids=scenes,
            download_meshes=not config.skip_meshes,
            download_atek=not config.skip_atek,
        )


if __name__ == "__main__":
    main()

"""Metadata management for ASE dataset.

Parses ASE download JSONs to map scenes → snippets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SceneInfo:
    """Scene with mesh + ATEK snippet information."""

    scene_id: str
    mesh_url: str
    mesh_sha: str
    snippet_ids: list[str]


class ASEMetadata:
    """Parse ASE JSONs to map scenes with meshes to their snippets."""

    def __init__(self, url_dir: Path):
        self.url_dir = Path(url_dir)
        self.scenes: dict[str, SceneInfo] = {}
        self._parse_jsons()

    def _parse_jsons(self) -> None:
        """Parse mesh and ATEK JSONs."""
        mesh_json = self.url_dir / "ase_mesh_download_urls.json"
        atek_json = self.url_dir / "AriaSyntheticEnvironment_ATEK_download_urls.json"

        # Parse meshes
        with mesh_json.open() as f:
            mesh_data = json.load(f)

        # Parse ATEK
        with atek_json.open() as f:
            atek_data = json.load(f)

        wds_urls = atek_data["atek_data_for_all_configs"]["efm"]["wds_file_urls"]

        # Build scene mapping
        for entry in mesh_data:
            scene_id = entry["filename"].replace("scene_ply_", "").replace(".zip", "")

            # Get snippets for this scene
            snippet_ids = []
            if scene_id in wds_urls:
                snippet_ids = [k.replace("_tar", "") for k in sorted(wds_urls[scene_id].keys())]

            self.scenes[scene_id] = SceneInfo(
                scene_id=scene_id,
                mesh_url=entry["cdn"],
                mesh_sha=entry["sha"],
                snippet_ids=snippet_ids,
            )

    def get_scenes(self, n: int | None = None, max_snippets: int | None = None) -> list[SceneInfo]:
        """Get N scenes with meshes and snippets.

        Args:
            n: Number of scenes to return (None = all)
            max_snippets: Maximum snippets per scene (None = all)

        Returns:
            List of SceneInfo sorted by scene_id
        """
        scenes = sorted(self.scenes.values(), key=lambda s: s.scene_id)

        # Limit snippets per scene if requested
        if max_snippets is not None:
            scenes = [
                SceneInfo(
                    scene_id=s.scene_id,
                    mesh_url=s.mesh_url,
                    mesh_sha=s.mesh_sha,
                    snippet_ids=s.snippet_ids[:max_snippets],
                )
                for s in scenes
            ]

        return scenes[:n] if n else scenes

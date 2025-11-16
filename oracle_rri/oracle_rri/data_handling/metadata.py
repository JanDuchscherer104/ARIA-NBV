"""Metadata management for ASE dataset download manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SceneMetadata:
    """Aggregated metadata for one ASE scene across ATEK configs."""

    scene_id: str
    has_gt_mesh: bool
    mesh_url: str | None
    mesh_sha: str | None
    snippet_count: int
    snippet_ids: list[str]
    atek_config: str
    total_frames: int


class ASEMetadata:
    """Parse mesh + ATEK URL JSONs to expose scene-level metadata."""

    def __init__(self, url_dir: Path):
        self.url_dir = Path(url_dir)
        if not self.url_dir.exists():
            raise FileNotFoundError("URL directory not found")
        self.mesh_scene_ids: set[str] = set()
        self.scenes: dict[str, SceneMetadata] = {}
        self._parse()

    def _parse(self) -> None:
        mesh_json = self.url_dir / "ase_mesh_download_urls.json"
        atek_json = self.url_dir / "AriaSyntheticEnvironment_ATEK_download_urls.json"
        mesh_data = json.load(mesh_json.open()) if mesh_json.exists() else []
        atek_data = json.load(atek_json.open()) if atek_json.exists() else {"atek_data_for_all_configs": {}}

        mesh_lookup: dict[str, tuple[str, str]] = {}
        for entry in mesh_data:
            scene_id = entry["filename"].replace("scene_ply_", "").replace(".zip", "")
            mesh_lookup[scene_id] = (entry["cdn"], entry["sha"])
            self.mesh_scene_ids.add(scene_id)

        configs = atek_data.get("atek_data_for_all_configs", {})
        for cfg_name, cfg in configs.items():
            for seq in cfg.get("sequences", []):
                scene_id = seq["sequence_name"]
                tar_urls: List[str] = seq.get("tar_urls", [])
                snippet_ids = [f"{scene_id}_seq_{i:03d}" for i, _ in enumerate(tar_urls)]
                has_mesh = scene_id in mesh_lookup
                mesh_url, mesh_sha = mesh_lookup.get(scene_id, (None, None))
                total_frames = seq.get("num_frames", 0)
                sm = SceneMetadata(
                    scene_id=scene_id,
                    has_gt_mesh=has_mesh,
                    mesh_url=mesh_url,
                    mesh_sha=mesh_sha,
                    snippet_count=len(tar_urls),
                    snippet_ids=snippet_ids,
                    atek_config=cfg_name,
                    total_frames=total_frames,
                )
                self.scenes[scene_id] = sm

    def get_scenes_with_meshes(self) -> list[SceneMetadata]:
        return [s for s in self.scenes.values() if s.has_gt_mesh]

    def filter_scenes(self, min_snippets: int = 0, require_mesh: bool = False, config: str | None = None) -> list[SceneMetadata]:
        scenes = list(self.scenes.values())
        if config is not None:
            scenes = [s for s in scenes if s.atek_config == config]
        if require_mesh:
            scenes = [s for s in scenes if s.has_gt_mesh]
        scenes = [s for s in scenes if s.snippet_count >= min_snippets]
        return scenes

    def get_scenes(self, n: int | None = None, max_snippets: int | None = None) -> list[SceneMetadata]:
        scenes = sorted(self.scenes.values(), key=lambda s: s.scene_id)
        if max_snippets is not None:
            scenes = [
                SceneMetadata(
                    scene_id=s.scene_id,
                    has_gt_mesh=s.has_gt_mesh,
                    mesh_url=s.mesh_url,
                    mesh_sha=s.mesh_sha,
                    snippet_count=min(s.snippet_count, max_snippets),
                    snippet_ids=s.snippet_ids[:max_snippets],
                    atek_config=s.atek_config,
                    total_frames=s.total_frames,
                )
                for s in scenes
            ]
        return scenes[:n] if n else scenes


__all__ = ["SceneMetadata", "ASEMetadata"]

import torch
import trimesh

from oracle_rri.configs.path_config import PathConfig
from oracle_rri.data.efm_views import EfmSnippetView
from oracle_rri.data.mesh_cache import MeshProcessSpec, mesh_from_snippet


def _make_sample(mesh: trimesh.Trimesh, spec: MeshProcessSpec) -> EfmSnippetView:
    bounds_min = torch.tensor(spec.bounds_min)
    bounds_max = torch.tensor(spec.bounds_max)
    return EfmSnippetView(
        efm={},
        scene_id=spec.scene_id,
        snippet_id=spec.snippet_id or "snippet",
        mesh=mesh,
        crop_bounds=(bounds_min, bounds_max),
        mesh_cache_key=spec.hash(),
        mesh_specs=spec,
    )


def test_mesh_from_snippet_persists_and_reuses_cache(tmp_path):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    spec = MeshProcessSpec(
        scene_id="123",
        snippet_id="abc",
        bounds_min=[-0.6, -0.6, -0.6],
        bounds_max=[0.6, 0.6, 0.6],
        margin_m=0.0,
        simplify_ratio=None,
        max_faces=None,
        crop_min_keep_ratio=0.05,
    )

    paths = PathConfig()
    original_processed = paths.processed_meshes
    paths_processed = tmp_path / "processed"
    paths_processed.mkdir(parents=True, exist_ok=True)

    try:
        paths.processed_meshes = paths_processed
        sample = _make_sample(mesh, spec)

        first = mesh_from_snippet(sample, paths=paths, want_p3d=True)
        assert first.processed.path.exists()
        assert first.p3d is not None
        assert first.processed.cache_hit is False

        second = mesh_from_snippet(sample, paths=paths, want_p3d=True)
        assert second.processed.cache_hit is True
        assert second.processed.path.exists()
    finally:
        # restore singleton to avoid side-effects
        paths.processed_meshes = original_processed

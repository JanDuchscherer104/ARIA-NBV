import torch
import trimesh

from oracle_rri.configs import PathConfig
from oracle_rri.data.mesh_cache import MeshProcessSpec, load_or_process_mesh
from oracle_rri.utils import Console


def test_processed_mesh_is_cached(tmp_path):
    console = Console.with_prefix("test_mesh_cache")
    path_cfg = PathConfig(root=tmp_path)

    mesh_raw = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    spec = MeshProcessSpec(
        scene_id="123",
        snippet_id="abc",
        bounds_min=[-1.0, -1.0, -1.0],
        bounds_max=[1.0, 1.0, 1.0],
        margin_m=0.1,
        simplify_ratio=0.5,
        max_faces=None,
        crop_min_keep_ratio=0.1,
    )

    processed_first = load_or_process_mesh(mesh_raw, spec, path_cfg, console=console)
    assert processed_first.path.exists()
    faces_first = processed_first.faces.shape[0]
    assert faces_first > 0
    assert not processed_first.cache_hit

    # second call should reuse cache and keep face count
    processed_second = load_or_process_mesh(mesh_raw, spec, path_cfg, console=console)
    assert processed_second.cache_hit
    assert processed_second.faces.shape[0] == faces_first
    assert torch.allclose(processed_second.verts.float(), processed_first.verts.float())

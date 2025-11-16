"""Integration test that exercises real ASE data and depth-to-mesh alignment."""

from pathlib import Path

import pytest

from oracle_rri.analysis import DepthDebuggerConfig
from oracle_rri.utils import Console


@pytest.mark.integration
def test_depth_projection_real_sample():
    """Load one real snippet, project depth to 3D, and measure distance to GT mesh."""
    scene_id = "81056"
    shard_dir = Path(".data/ase_efm") / scene_id
    mesh_path = Path(".data/ase_meshes") / f"scene_ply_{scene_id}.ply"

    if not shard_dir.exists() or not mesh_path.exists():
        pytest.skip("Real ASE shards or mesh not available locally.")

    config = DepthDebuggerConfig(
        scene_id=scene_id,
        tar_glob=str(shard_dir / "shards-*.tar"),
        mesh_path=mesh_path,
        max_points=400,
        mesh_simplify_ratio=0.2,
        mesh_vertex_cap=3000,
        device="cpu",
        verbose=True,
    )

    debugger = config.setup_target()
    result = debugger.run()

    console = Console.with_prefix("test_depth_projection_real_sample")
    console.plog(result)

    assert result.num_points > 0
    assert result.mean >= 0.0

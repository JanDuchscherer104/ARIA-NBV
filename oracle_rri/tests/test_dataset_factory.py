from __future__ import annotations

from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ATEK_PATH = REPO_ROOT / "external" / "ATEK"
if str(ATEK_PATH) not in sys.path:
    sys.path.insert(0, str(ATEK_PATH))

from oracle_rri.config import DatasetPaths, OracleConfig
from oracle_rri.data.factory import DatasetFactory


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> DatasetPaths:
    """Create a minimal dataset layout for testing the factory."""

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "atek.json").write_text("{}")
    (manifests_dir / "mesh.json").write_text("{}")

    wds_root = tmp_path / "wds"
    shard_dir = wds_root / "1001"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shards-0000.tar").touch()

    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    (mesh_root / "scene_ply_1001.ply").write_text("ply\n")

    (wds_root / "local_validation_tars.yaml").write_text("tars: {}\n")
    (wds_root / "local_all_tars.yaml").write_text("tars: {}\n")
    (wds_root / "local_train_tars.yaml").write_text(
        "tars:\n  '1001':\n    - 1001/shards-0000.tar\n"
    )

    return DatasetPaths(
        atek_manifest=manifests_dir / "atek.json",
        mesh_manifest=manifests_dir / "mesh.json",
        wds_root=wds_root,
        mesh_root=mesh_root,
    )


def test_dataset_factory_dataframe_contains_split(tmp_dataset: DatasetPaths, tmp_path: Path) -> None:
    config = OracleConfig(dataset=tmp_dataset, output_root=tmp_path / "outputs")
    factory = DatasetFactory(config)

    df = factory.dataframe()
    assert not df.empty
    assert set(df.columns) == {"scene_id", "split", "shard_path", "mesh_path"}
    assert df.loc[0, "split"] == "train"
    assert Path(df.loc[0, "shard_path"]).exists()
    assert Path(df.loc[0, "mesh_path"]).name == "scene_ply_1001.ply"

    shards = factory.list_shards()
    assert shards == [Path(df.loc[0, "shard_path"])]

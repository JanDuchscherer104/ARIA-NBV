"""Pytest fixtures for data handling tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_url_dir(tmp_path: Path) -> Path:
    """Create temporary directory with mock download URL JSONs.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to temporary URL directory with mock JSONs
    """
    url_dir = tmp_path / "aria_download_urls"
    url_dir.mkdir()
    return url_dir


@pytest.fixture
def mock_mesh_urls_json(tmp_url_dir: Path) -> Path:
    """Create mock ase_mesh_download_urls.json.

    Returns:
        Path to created JSON file
    """
    # Match actual ASE format: list of entries with filename, cdn, sha
    mesh_data = [
        {
            "filename": "scene_ply_82832.zip",
            "cdn": "https://example.com/scene_ply_82832.ply.zip",
            "sha": "abc123def456",
        },
        {
            "filename": "scene_ply_81022.zip",
            "cdn": "https://example.com/scene_ply_81022.ply.zip",
            "sha": "def789ghi012",
        },
        {
            "filename": "scene_ply_80001.zip",
            "cdn": "https://example.com/scene_ply_80001.ply.zip",
            "sha": "ghi345jkl678",
        },
    ]

    mesh_urls_path = tmp_url_dir / "ase_mesh_download_urls.json"
    with open(mesh_urls_path, "w") as f:
        json.dump(mesh_data, f)

    return mesh_urls_path


@pytest.fixture
def mock_atek_urls_json(tmp_url_dir: Path) -> Path:
    """Create mock AriaSyntheticEnvironment_ATEK_download_urls.json.

    Returns:
        Path to created JSON file
    """
    atek_data = {
        "atek_data_for_all_configs": {
            "efm": {
                "sequences": [
                    {
                        "sequence_name": "82832",
                        "tar_urls": ["url1.tar", "url2.tar", "url3.tar"],
                        "num_frames": 300,
                    },
                    {
                        "sequence_name": "81022",
                        "tar_urls": ["url1.tar", "url2.tar"],
                        "num_frames": 100,
                    },
                    {
                        "sequence_name": "90000",
                        "tar_urls": ["url1.tar"],
                        "num_frames": 200,
                    },
                ],
            },
            "cubercnn": {
                "sequences": [
                    {
                        "sequence_name": "82832",
                        "tar_urls": ["url1.tar"],
                        "num_frames": 100,
                    },
                ],
            },
        }
    }

    atek_urls_path = tmp_url_dir / "AriaSyntheticEnvironment_ATEK_download_urls.json"
    with open(atek_urls_path, "w") as f:
        json.dump(atek_data, f)

    return atek_urls_path


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for downloads.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_mesh_file(tmp_output_dir: Path) -> Path:
    """Create mock mesh PLY file.

    Args:
        tmp_output_dir: Temporary output directory

    Returns:
        Path to mock PLY file
    """
    mesh_dir = tmp_output_dir / "ase_meshes"
    mesh_dir.mkdir()

    mesh_path = mesh_dir / "scene_ply_82832.ply"

    # Create minimal valid PLY file
    ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
3 0 1 2
"""
    mesh_path.write_text(ply_content)

    return mesh_path

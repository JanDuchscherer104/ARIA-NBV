"""Tests for utils.py module."""

from pathlib import Path

import pytest

from aria_nbv.utils.viz_utils import (
    extract_scene_id_from_sequence_name,
    validate_scene_data,
)


class TestExtractSceneId:
    """Tests for extract_scene_id_from_sequence_name."""

    def test_valid_sequence_name(self):
        """Test extracting scene ID from valid sequence name."""
        assert extract_scene_id_from_sequence_name("82832_seq_000") == "82832"
        assert extract_scene_id_from_sequence_name("81022_seq_001") == "81022"
        assert extract_scene_id_from_sequence_name("90000_seq_999") == "90000"

    def test_sequence_name_without_prefix(self):
        """Test sequence name without '_seq_' prefix."""
        # Should still extract first part
        assert extract_scene_id_from_sequence_name("82832") == "82832"
        assert extract_scene_id_from_sequence_name("12345_other_suffix") == "12345"

    def test_empty_string(self):
        """Test empty string."""
        assert extract_scene_id_from_sequence_name("") == ""

    def test_numeric_only(self):
        """Test numeric-only scene ID."""
        assert extract_scene_id_from_sequence_name("12345") == "12345"


class TestValidateSceneData:
    """Tests for validate_scene_data."""

    def test_validate_with_mesh(
        self,
        tmp_path: Path,
        mock_mesh_file: Path,
    ):
        """Test validation when mesh exists."""
        data_dir = mock_mesh_file.parent.parent

        result = validate_scene_data(
            scene_id="82832",
            data_dir=data_dir,
            require_mesh=True,
            require_atek=False,
        )

        assert result["mesh_exists"] is True
        assert result["mesh_path"] == mock_mesh_file

    def test_validate_without_mesh(self, tmp_path: Path):
        """Test validation when mesh doesn't exist."""
        result = validate_scene_data(
            scene_id="nonexistent",
            data_dir=tmp_path,
            require_mesh=False,
            require_atek=False,
        )

        assert result["mesh_exists"] is False
        assert result["mesh_path"] is None

    def test_validate_missing_required_mesh(self, tmp_path: Path):
        """Test validation fails when mesh required but missing."""
        with pytest.raises(FileNotFoundError, match="GT mesh not found"):
            validate_scene_data(
                scene_id="nonexistent",
                data_dir=tmp_path,
                require_mesh=True,
                require_atek=False,
            )

    def test_validate_with_atek_data(self, tmp_path: Path):
        """Test validation when ATEK data exists."""
        # Create mock ATEK directory
        scene_id = "82832"
        atek_dir = tmp_path / "ase_atek" / scene_id
        atek_dir.mkdir(parents=True)
        (atek_dir / "shard-0.tar").touch()

        result = validate_scene_data(
            scene_id=scene_id,
            data_dir=tmp_path,
            require_mesh=False,
            require_atek=True,
        )

        assert result["atek_exists"] is True
        assert result["atek_path"] == atek_dir

    def test_validate_missing_required_atek(self, tmp_path: Path):
        """Test validation fails when ATEK required but missing."""
        with pytest.raises(FileNotFoundError, match="ATEK data not found"):
            validate_scene_data(
                scene_id="nonexistent",
                data_dir=tmp_path,
                require_mesh=False,
                require_atek=True,
            )

    def test_validate_all_data_exists(
        self,
        tmp_path: Path,
        mock_mesh_file: Path,
    ):
        """Test validation when both mesh and ATEK exist."""
        data_dir = mock_mesh_file.parent.parent
        scene_id = "82832"

        # Create ATEK directory
        atek_dir = data_dir / "ase_atek" / scene_id
        atek_dir.mkdir(parents=True)
        (atek_dir / "shard-0.tar").touch()

        result = validate_scene_data(
            scene_id=scene_id,
            data_dir=data_dir,
            require_mesh=True,
            require_atek=True,
        )

        assert result["mesh_exists"] is True
        assert result["atek_exists"] is True
        assert result["mesh_path"] == mock_mesh_file
        assert result["atek_path"] == atek_dir

    def test_validate_optional_data(self, tmp_path: Path):
        """Test validation when no data required."""
        result = validate_scene_data(
            scene_id="any_scene",
            data_dir=tmp_path,
            require_mesh=False,
            require_atek=False,
        )

        # Should succeed even with missing data
        assert result["mesh_exists"] is False
        assert result["atek_exists"] is False

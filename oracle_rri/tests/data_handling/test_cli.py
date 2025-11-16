"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from oracle_rri.data_handling.cli import CLIDownloaderSettings, main


class TestCLIDownloaderSettings:
    """Tests for CLIDownloaderSettings."""

    def test_default_values(self):
        """Test default settings values."""
        with patch("sys.argv", ["cli"]):
            settings = CLIDownloaderSettings()

            assert settings.url_dir == Path(".data/aria_download_urls")
            assert settings.output_dir == Path(".data")
            assert settings.verbose is True
            assert settings.all_with_meshes is False

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        with patch(
            "sys.argv",
            ["cli", "--output-dir", "/tmp/output", "--verbose", "false"],
        ):
            settings = CLIDownloaderSettings()

            assert settings.output_dir == Path("/tmp/output")
            assert settings.verbose is False

    def test_scene_ids_parsing(self):
        """Test parsing scene ID list."""
        with patch(
            "sys.argv",
            ["cli", "--scene-ids", "82832", "81022", "90000"],
        ):
            settings = CLIDownloaderSettings()

            assert settings.scene_ids == ["82832", "81022", "90000"]


class TestCLIMain:
    """Tests for CLI main function."""

    @patch("oracle_rri.data_handling.cli.ASEDownloaderConfig")
    @patch("oracle_rri.data_handling.cli.CLIDownloaderSettings")
    def test_main_all_with_meshes(
        self,
        mock_settings_class: MagicMock,
        mock_config_class: MagicMock,
    ):
        """Test main with --all-with-meshes flag."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.all_with_meshes = True
        mock_settings.scene_ids = None
        mock_settings.config_path = None
        mock_settings.min_snippets = 5
        mock_settings.config = "efm"
        mock_settings.overwrite = False
        mock_settings.url_dir = Path(".data/urls")
        mock_settings.output_dir = Path(".data/output")
        mock_settings.metadata_cache_path = None
        mock_settings.verbose = True
        mock_settings_class.return_value = mock_settings

        mock_config = MagicMock()
        mock_downloader = MagicMock()
        mock_config.setup_target.return_value = mock_downloader
        mock_config_class.return_value = mock_config

        # Run main
        main()

        # Verify download_scenes_with_meshes was called
        mock_downloader.download_scenes_with_meshes.assert_called_once_with(
            min_snippets=5,
            config="efm",
            overwrite=False,
        )

    @patch("oracle_rri.data_handling.cli.ASEDownloaderConfig")
    @patch("oracle_rri.data_handling.cli.CLIDownloaderSettings")
    def test_main_specific_scenes(
        self,
        mock_settings_class: MagicMock,
        mock_config_class: MagicMock,
    ):
        """Test main with specific scene IDs."""
        mock_settings = MagicMock()
        mock_settings.all_with_meshes = False
        mock_settings.scene_ids = ["82832", "81022"]
        mock_settings.config_path = None
        mock_settings.meshes_only = False
        mock_settings.atek_only = False
        mock_settings.config = "efm"
        mock_settings.overwrite = False
        mock_settings.url_dir = Path(".data/urls")
        mock_settings.output_dir = Path(".data/output")
        mock_settings.metadata_cache_path = None
        mock_settings.verbose = True
        mock_settings_class.return_value = mock_settings

        mock_config = MagicMock()
        mock_downloader = MagicMock()
        mock_config.setup_target.return_value = mock_downloader
        mock_config_class.return_value = mock_config

        main()

        # Verify download_scenes was called
        mock_downloader.download_scenes.assert_called_once()
        call_kwargs = mock_downloader.download_scenes.call_args[1]
        assert call_kwargs["scene_ids"] == ["82832", "81022"]

    @patch("oracle_rri.data_handling.cli.ASEDownloaderConfig")
    @patch("oracle_rri.data_handling.cli.CLIDownloaderSettings")
    def test_main_meshes_only(
        self,
        mock_settings_class: MagicMock,
        mock_config_class: MagicMock,
    ):
        """Test main with --meshes-only flag."""
        mock_settings = MagicMock()
        mock_settings.all_with_meshes = False
        mock_settings.scene_ids = ["82832"]
        mock_settings.config_path = None
        mock_settings.meshes_only = True
        mock_settings.overwrite = False
        mock_settings.url_dir = Path(".data/urls")
        mock_settings.output_dir = Path(".data/output")
        mock_settings.metadata_cache_path = None
        mock_settings.verbose = True
        mock_settings_class.return_value = mock_settings

        mock_config = MagicMock()
        mock_downloader = MagicMock()
        mock_config.setup_target.return_value = mock_downloader
        mock_config_class.return_value = mock_config

        main()

        # Verify download_meshes was called
        mock_downloader.download_meshes.assert_called_once_with(
            scene_ids=["82832"],
            overwrite=False,
        )

    @patch("oracle_rri.data_handling.cli.ASEDownloaderConfig")
    @patch("oracle_rri.data_handling.cli.CLIDownloaderSettings")
    @patch("builtins.print")
    def test_main_no_action(
        self,
        mock_print: MagicMock,
        mock_settings_class: MagicMock,
        mock_config_class: MagicMock,
    ):
        """Test main with no action specified."""
        mock_settings = MagicMock()
        mock_settings.all_with_meshes = False
        mock_settings.scene_ids = None
        mock_settings.config_path = None
        mock_settings.url_dir = Path(".data/urls")
        mock_settings.output_dir = Path(".data/output")
        mock_settings.metadata_cache_path = None
        mock_settings.verbose = True
        mock_settings_class.return_value = mock_settings

        mock_config = MagicMock()
        mock_downloader = MagicMock()
        mock_config.setup_target.return_value = mock_downloader
        mock_config_class.return_value = mock_config

        main()

        # Verify warning message printed
        mock_print.assert_called()
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("No download action specified" in str(call) for call in print_calls)

    @patch("oracle_rri.data_handling.cli.ASEDownloaderConfig")
    @patch("oracle_rri.data_handling.cli.CLIDownloaderSettings")
    def test_main_with_config_path(
        self,
        mock_settings_class: MagicMock,
        mock_config_class: MagicMock,
        tmp_path: Path,
    ):
        """Test main with config file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[oracle_rri.data_handling.downloader]\nverbose = true\n")

        mock_settings = MagicMock()
        mock_settings.config_path = config_file
        mock_settings.all_with_meshes = True
        mock_settings.min_snippets = 0
        mock_settings.config = "efm"
        mock_settings.overwrite = False
        mock_settings_class.return_value = mock_settings

        mock_loaded_config = MagicMock()
        mock_loaded_config.config = "efm"
        mock_downloader = MagicMock()
        mock_loaded_config.setup_target.return_value = mock_downloader

        mock_config_class.from_toml.return_value = mock_loaded_config

        main()

        # Verify config loaded from file
        mock_config_class.from_toml.assert_called_once_with(config_file)
        mock_downloader.download_scenes_with_meshes.assert_called_once()

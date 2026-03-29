# Data Handling Tests

Comprehensive test suite for the `oracle_rri.data_handling` module.

## Test Structure

```
tests/data_handling/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures
├── test_metadata.py         # Metadata parsing tests
├── test_downloader.py       # Downloader tests
├── test_dataset.py          # PyTorch dataset tests
├── test_utils.py            # Utility function tests
└── test_cli.py              # CLI tests
```

## Running Tests

### Run All Tests

```bash
# From repository root
cd oracle_rri
pytest tests/data_handling/ -v

# With coverage
pytest tests/data_handling/ --cov=oracle_rri.data_handling --cov-report=html
```

### Run Specific Test Files

```bash
# Test metadata parsing
pytest tests/data_handling/test_metadata.py -v

# Test downloader
pytest tests/data_handling/test_downloader.py -v

# Test dataset
pytest tests/data_handling/test_dataset.py -v

# Test CLI
pytest tests/data_handling/test_cli.py -v
```

### Run Specific Test Classes or Functions

```bash
# Test specific class
pytest tests/data_handling/test_metadata.py::TestASEMetadata -v

# Test specific function
pytest tests/data_handling/test_metadata.py::TestASEMetadata::test_parse_mesh_urls -v
```

## Test Coverage

### `test_metadata.py` (21 tests)

Tests for metadata parsing and management:

- **SceneMetadata**: Creation, validation
- **ASEMetadata**:
  - Parsing mesh URLs
  - Parsing ATEK URLs
  - Scene filtering (by snippets, mesh availability, config)
  - Save/load caching
  - Error handling

### `test_downloader.py` (14 tests)

Tests for download orchestration:

- **ASEDownloaderConfig**:
  - Configuration creation
  - Path resolution
  - Config-as-Factory pattern
- **ASEDownloader**:
  - Mesh download with SHA validation
  - Scene filtering
  - Directory creation
  - Error handling

### `test_dataset.py`

Tests for the typed `ASEDataset`:

- Config validation (missing tar URLs, autofill from `scene_ids`)
- Typed sample wrapping (`CameraStream`, `Trajectory`, `SemiDensePoints`)
- EFM3D remapping via `to_efm_dict`
- DataLoader integration with `ase_collate`
- Mesh loading and caching

### `test_utils.py` (9 tests)

Tests for utility functions:

- **extract_scene_id_from_sequence_name**: Various input formats
- **validate_scene_data**: Mesh/ATEK validation, error handling

### `test_cli.py` (8 tests)

Tests for CLI interface:

- **CLIDownloaderSettings**: Argument parsing
- **main()**:
  - --all-with-meshes flag
  - --scene-ids flag
  - --meshes-only flag
  - Config file loading
  - Error messages

## Test Fixtures

Located in `conftest.py`:

- **`tmp_url_dir`**: Temporary directory for mock download URLs
- **`tmp_output_dir`**: Temporary output directory
- **`mock_mesh_urls_json`**: Mock mesh download URLs (3 scenes)
- **`mock_atek_urls_json`**: Mock ATEK download URLs (3 scenes, various configs)
- **`mock_mesh_file`**: Minimal valid PLY mesh file

## Mock Data Structure

### Mesh URLs (3 scenes with GT meshes)
- 82832: Has mesh + 3 efm snippets + 1 cubercnn snippet
- 81022: Has mesh + 2 efm snippets
- 80001: Has mesh (no ATEK data in fixture)

### ATEK URLs (3 scenes)
- 82832: 3 efm snippets (300 frames), 1 cubercnn snippet (100 frames)
- 81022: 2 efm snippets (100 frames)
- 90000: 1 efm snippet (200 frames) - NO MESH

This setup allows testing:
- Scenes with meshes + ATEK data (82832, 81022)
- Scenes with ATEK but no mesh (90000)
- Scenes with meshes but no ATEK (80001)
- Multiple configs for same scene (82832)

## Test Categories

### Unit Tests
Most tests are unit tests with mocked dependencies:
- `test_metadata.py`: Pure unit tests (no mocking needed)
- `test_downloader.py`: Mocks HTTP requests and file I/O
- `test_dataset.py`: Mocks ATEK loader
- `test_utils.py`: Pure unit tests
- `test_cli.py`: Mocks all dependencies

### Integration Tests
Located in `*Integration` test classes:
- `TestASEDownloaderIntegration`: Config → Downloader → Metadata flow
- `TestASESceneDatasetIntegration`: Config → Dataset → Sample flow

## Common Test Patterns

### Config-as-Factory Testing

```python
def test_config_to_instance():
    config = ASEDownloaderConfig(mode="download", url_dir=Path(".data/urls"))
    instance = config.setup_target()
    assert isinstance(instance, ASEDownloader)
```

### Mocking External Dependencies

```python
@patch("oracle_rri.data_handling.downloader.requests.get")
def test_download(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    # Test logic...
```

### Using Fixtures

```python
def test_with_fixtures(
    tmp_url_dir: Path,
    mock_mesh_urls_json: Path,
):
    metadata = ASEMetadata(tmp_url_dir)
    assert len(metadata.mesh_scene_ids) == 3
```

## Expected Test Output

```
tests/data_handling/test_metadata.py::TestSceneMetadata::test_creation PASSED
tests/data_handling/test_metadata.py::TestSceneMetadata::test_no_mesh PASSED
tests/data_handling/test_metadata.py::TestASEMetadata::test_parse_mesh_urls PASSED
...
tests/data_handling/test_cli.py::TestCLIMain::test_main_with_config_path PASSED

======================== 68 passed in 2.34s =========================
```

## Adding New Tests

When adding new functionality:

1. **Add fixtures** to `conftest.py` if needed
2. **Create test class** in appropriate test file
3. **Follow naming convention**: `test_<functionality>`
4. **Use mocking** for external dependencies
5. **Test both success and failure** cases
6. **Document** expected behavior in docstrings

Example:

```python
class TestNewFeature:
    """Tests for new feature."""

    def test_success_case(self):
        """Test successful execution."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        assert expected == actual

    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ExpectedError):
            # Act that should raise
            ...
```

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure package installed in editable mode
cd oracle_rri
pip install -e .
```

### Fixtures Not Found

Check that `conftest.py` is in the correct directory and properly imported.

### Mock Not Working

Verify the patch target path matches the actual import in the tested module:
```python
# If module does: from oracle_rri.utils import Console
# Patch as: @patch("oracle_rri.data_handling.downloader.Console")
```

## Continuous Integration

These tests are designed to run in CI environments:

- No external dependencies (all mocked)
- Use temporary directories (pytest `tmp_path`)
- Fast execution (< 5 seconds)
- Clear failure messages

## Future Test Additions

- [ ] ATEK WDS download integration tests (once implemented)
- [ ] Proper snippet-level indexing tests (once implemented)
- [ ] Performance benchmarks for large datasets
- [ ] End-to-end notebook integration tests

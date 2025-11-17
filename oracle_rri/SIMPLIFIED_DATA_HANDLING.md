# Simplified ASE Data Handling

## Overview

Stripped down to essentials:
- **metadata.py** (82 lines): Parse JSONs → scene→snippet mapping
- **downloader.py** (163 lines): Download N meshes + their snippets
- **cli.py** (151 lines): CLI using pydantic-settings

**Total: ~400 lines** (was 1000+)

## Usage

### CLI Examples

```bash
# List available scenes
python -m oracle_rri.data_handling.cli list -n 10

# Download 5 scenes, max 2 snippets each
python -m oracle_rri.data_handling.cli --n_scenes=5 --max_snippets=2

# Download all 100 scenes with all snippets
python -m oracle_rri.data_handling.cli --n_scenes=0

# Only download meshes (skip ATEK data)
python -m oracle_rri.data_handling.cli --n_scenes=3 --skip_atek

# See all options
python -m oracle_rri.data_handling.cli --help
```

### Programmatic Usage

```python
from pathlib import Path
from oracle_rri.data_handling import ASEDownloaderConfig

# Create downloader
config = ASEDownloaderConfig(mode="download", verbose=True)
downloader = config.setup_target()

# Get 5 scenes, max 2 snippets each
scenes = downloader.metadata.get_scenes(n=5, max_snippets=2)

# Download
downloader.download_scenes(
    scenes=scenes,
    download_meshes=True,
    download_atek=True,
)
```

## Architecture

### metadata.py

- **SceneInfo**: Simple dataclass (scene_id, mesh_url, mesh_sha, snippet_ids)
- **ASEMetadata**: Parses both JSONs in `__init__`, stores dict[scene_id → SceneInfo]
- **get_scenes(n, max_snippets)**: Returns filtered/limited scenes

### downloader.py

- **ASEDownloader**: Wraps ATEK download tools
- **download_scenes(scenes)**: Orchestrates mesh + ATEK downloads
- **_download_meshes()**: Calls ATEK's ase_mesh_downloader
- **_download_atek()**: Filters ATEK JSON, calls atek_wds_data_downloader

### cli.py

- **DownloadCLI**: pydantic-settings for download command
- **ListCLI**: pydantic-settings for list command
- **main()**: Subcommand dispatcher (download/list)

## What Was Removed

- ❌ Validation methods (validate_scene_completeness)
- ❌ Filtering methods (filter_scenes, get_scenes_with_meshes)
- ❌ Metadata caching (save/load)
- ❌ Manual mesh download fallback
- ❌ Complex scene selection logic
- ❌ Reverse snippet→scene mapping
- ❌ Frame counting
- ❌ Utils module (extract_scene_id, validate_scene_data)

## What Remains

- ✓ Parse mesh + ATEK JSONs
- ✓ Map scenes → snippets
- ✓ Download N scenes with max K snippets
- ✓ Simple CLI with pydantic-settings
- ✓ Clean Config-as-Factory pattern

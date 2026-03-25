# Downloader list-mode: total snippet count

## Goal

`nbv-downloader -m list` printed the scene list but did not show the total number of snippets across the listed scenes.

## What changed

- `oracle_rri/oracle_rri/data/downloader.py`
  - `cli_list()` now prints `Total snippets (all GT-mesh scenes): <N>` after the per-scene listing.
  - When `n` limits the list, it also prints `Total snippets (shown scenes): <N_shown>` to avoid ambiguity.

## Tests / validation

- `uv run ruff format oracle_rri/oracle_rri/data/downloader.py oracle_rri/tests/data_handling/test_downloader_cli_list_snippet_totals.py`
- `uv run ruff check oracle_rri/oracle_rri/data/downloader.py oracle_rri/tests/data_handling/test_downloader_cli_list_snippet_totals.py`
- `uv run pytest -q oracle_rri/tests/data_handling/test_downloader_cli_list_snippet_totals.py oracle_rri/tests/data_handling/test_metadata.py`

## Notes / suggestions

- `ASEDownloaderConfig.mode` uses alias `m` and currently does **not** accept `mode=...` programmatically (pydantic alias population). If this becomes annoying for non-CLI usage/tests, consider enabling name-based population in `model_config` (e.g. `populate_by_name=True`) or switching to `validation_alias=AliasChoices("mode", "m")` for symmetry with other fields.
- From repo root, `uv run nbv-downloader ...` may require `--project oracle_rri` depending on `uv` invocation context.


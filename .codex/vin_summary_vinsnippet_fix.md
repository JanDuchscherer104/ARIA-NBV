Summary:
- Hardened VIN summary/debug paths to handle VinSnippetView without accessing `.efm` directly.
- Added VinSnippetView-compatible summaries in experimental VIN v1/v2 models.
- Made VIN diagnostics helper more robust via duck-typed snippet detection.

Open items / suggestions:
- Consider mirroring the same VinSnippetView handling in any remaining summary/plot helpers that still assume `EfmSnippetView.efm`.
- If VIN v1 truly requires raw EFM inputs, add an explicit UI guard to disable summarize/plots when only VinSnippetView is available.

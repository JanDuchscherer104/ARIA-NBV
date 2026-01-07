## Offline stats review (2026-01-03)

### Findings (overcomplication / redundancy)
- `oracle_rri/oracle_rri/app/panels/offline_stats.py:120` duplicates JSONL parsing logic for counts (`_count_index`) even though `read_cache_index_entries` already exists; could reuse it to avoid two readers.
- `oracle_rri/oracle_rri/app/panels/offline_stats.py:161` repeats TOML + `AriaNBVExperimentConfig` resolution that also happens inside `_collect_offline_cache_stats` (different helper) → consider a single shared resolver in `offline_cache_utils`.
- `oracle_rri/oracle_rri/app/panels/offline_stats.py:265` stores `dataset_snippets_by_scene` in `coverage_cache` but never reads it later.
- Plot sections repeat nearly identical Matplotlib boilerplate (create fig/ax, set title/x/y, st.pyplot, plt.close) for many histograms; a small helper could reduce duplication.

### Pandas method-chaining opportunities
- `oracle_rri/oracle_rri/app/panels/offline_stats.py:425`:
  - `pd.DataFrame(report.as_rows()).sort_values(..., na_position="last")`
- `oracle_rri/oracle_rri/app/panels/offline_stats.py:808`:
  - `(backbone_df.groupby("field", as_index=False)[selected_cols].mean().sort_values(sort_choice, ascending=False))`

### Optional follow-ups
- Add `st.cache_data` for `scan_dataset_snippets` (keyed by tar path + mtime) to avoid rescans during UI tweaks.
- Extract `plot_hist_mpl(...)` helper to simplify repeated histogram rendering.

### Changes applied
- Reused `read_cache_index_entries` for cache counts and coverage, removed unused `dataset_snippets_by_scene` cache entry.
- Added small histogram helpers to collapse repeated Matplotlib blocks.
- Applied pandas method chaining for coverage table + backbone summary.

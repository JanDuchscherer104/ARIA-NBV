Here is a precise audit of every redundancy I found across the PR branch (`codex/data-handling-overhaul`) and the existing `main`-branch surface. I've grouped them by category.

---

## 1. Wholesale Type/Model Duplication: Two Parallel `data` vs `data_handling` Packages

The most glaring problem: the PR introduces `aria_nbv/data_handling/` but **does not remove the originals** from `aria_nbv/data/`. These files are now defined in two places simultaneously:

| Concept | `aria_nbv/data/` (old, on main) | `aria_nbv/data_handling/` (new, on PR branch) |
|---|---|---|
| EFM views / snippet types | [`efm_views.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/efm_views.py) | [`efm_views.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/efm_views.py) |
| EFM dataset | [`efm_dataset.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/efm_dataset.py) | [`efm_dataset.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/efm_dataset.py) |
| EFM snippet loader | [`efm_snippet_loader.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/efm_snippet_loader.py) | [`efm_snippet_loader.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/efm_snippet_loader.py) |
| VIN oracle types + collation | [`vin_oracle_types.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_oracle_types.py) | [`vin_oracle_types.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/vin_oracle_types.py) |
| VIN oracle datasets | [`vin_oracle_datasets.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_oracle_datasets.py) | [`vin_oracle_datasets.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/vin_oracle_datasets.py) |
| Oracle cache (main logic) | [`offline_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/offline_cache.py) (40 KB) | [`oracle_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/oracle_cache.py) (36 KB) |
| VIN snippet cache | [`vin_snippet_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_snippet_cache.py) (34 KB) | [`vin_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/vin_cache.py) (36 KB) |
| VIN snippet provider | [`vin_snippet_provider.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_snippet_provider.py) | [`vin_provider.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/vin_provider.py) |
| Offline cache store helpers | [`offline_cache_store.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/offline_cache_store.py) | [`offline_cache_store.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/offline_cache_store.py) |
| Mesh cache | [`mesh_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/mesh_cache.py) | [`mesh_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/mesh_cache.py) |

**Every one of these is a live duplicate.** The old `aria_nbv/data/` files remain intact and are still imported by `app/`, `lightning/`, `vin/` etc. on `main`.

---

## 2. Type Definitions Scattered Across Multiple Files (Should Be One Place)

### `VinOracleBatch` + all its collation helpers
Defined in full (715 lines, including `_pad_1d`, `_pad_candidate_poses`, `_pad_trajectory`, `_pad_points`, `_stack_reference_poses`, `_stack_p3d_cameras`, `_stack_backbone_outputs`, `_gather_candidate`, `_shuffle_camera_param`) in **both**:
- [`aria_nbv/data/vin_oracle_types.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_oracle_types.py)
- [`aria_nbv/data_handling/vin_oracle_types.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/vin_oracle_types.py)

These are **identical logic**. There must be exactly one.

### `OracleRriCacheMetadata`, `OracleRriCacheEntry`, `OracleRriCacheSample`
Defined in [`aria_nbv/data/offline_cache_types.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/offline_cache_types.py), then re-implemented inside both `data_handling/oracle_cache.py` and `data_handling/cache_contracts.py` + `cache_index.py`. Three locations.

### `VinSnippetCacheMetadata`, `VinSnippetCacheEntry`, `VinSnippetCacheBuildResult`
Defined in [`aria_nbv/data/vin_snippet_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_snippet_cache.py) and again in `data_handling/vin_cache.py`. Two locations.

### `MeshProcessSpec`, `ProcessedMesh`
Defined in [`aria_nbv/data/mesh_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/mesh_cache.py) and again in [`aria_nbv/data_handling/mesh_cache.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/codex/data-handling-overhaul/aria_nbv/aria_nbv/data_handling/mesh_cache.py). Two locations.

### `SceneCoverage`, `CacheCoverageReport`
Defined in [`aria_nbv/data/offline_cache_coverage.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/offline_cache_coverage.py) — but the PR's `data_handling` layer has no equivalent, meaning coverage logic is orphaned in the old package while everything else migrates.

---

## 3. Duplicate `utils.py` Files (Helper Functions NOT in the Shared Utils Module)

| File | What it contains |
|---|---|
| [`aria_nbv/data/utils.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/utils.py) | Tiny module with data-local helpers |
| [`aria_nbv/data/vin_snippet_utils.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_snippet_utils.py) | `build_vin_snippet_view`, `vin_snippet_cache_config_hash`, `empty_vin_snippet` |

`vin_snippet_utils.py` contains pure utility functions (`build_vin_snippet_view`, hashing, empty-stub factory) that are consumed by both `vin_snippet_cache.py` and `vin_snippet_provider.py`. These should live in `utils/` or a single shared submodule — not in a domain package. The PR does not address this.

---

## 4. Plotting: 6+ Separate `plotting.py` Files With Duplicated Primitives

This is the single largest source of AI slop in the repo. The following private helpers appear independently in **multiple** plotting modules:

| Helper | Lives in |
|---|---|
| `_to_numpy` | `utils/plotting.py`, `vin/plotting.py` |
| `_plot_slice_grid` | `utils/plotting.py`, `vin/plotting.py` |
| `_histogram_overlay` | `utils/plotting.py`, `vin/plotting.py`, `rri_metrics/plotting.py` |
| `_plot_hist_counts_mpl` | `utils/plotting.py`, `rri_metrics/plotting.py` |
| `_pretty_label` | `vin/experimental/plotting.py`, `app/panels/common.py` |
| `_scalar_to_rgb` | `utils/plotting.py` only — but re-imported inconsistently |

The canonical home is [`aria_nbv/utils/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/utils/plotting.py). Every other module should import from there and delete its local copy. This is already partially acknowledged in the agent archive at [`.agents/archive/codex-legacy/flat/panels_helpers_audit.md`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/.agents/archive/codex-legacy/flat/panels_helpers_audit.md) — but the fix was **never applied**.

The full list of `plotting.py` files that need consolidation:
1. [`aria_nbv/utils/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/utils/plotting.py) ← the one true home
2. [`aria_nbv/data/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/plotting.py) (40 KB — domain-specific but borrows primitives)
3. [`aria_nbv/vin/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/vin/plotting.py) (73 KB — re-exports `_to_numpy`, `_histogram_overlay`, etc. in `__all__`)
4. [`aria_nbv/vin/experimental/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/vin/experimental/plotting.py) — has its own `_pretty_label`, `_pca_2d`
5. [`aria_nbv/pose_generation/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/pose_generation/plotting.py)
6. [`aria_nbv/rendering/plotting.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/rendering/plotting.py)

---

## 5. `OracleRriCacheConfig` / `VinSnippetCacheConfig` — Duplicated `_resolve_cache_dir` Validators

Both [`offline_cache.py:OracleRriCacheConfig`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/offline_cache.py#L69) and [`vin_snippet_cache.py:VinSnippetCacheConfig`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/data/vin_snippet_cache.py#L78) contain a verbatim-identical `@field_validator("cache_dir", mode="before")` method that resolves paths using `PathConfig`. This is copy-paste. It should be a single base class or a module-level helper in `utils/`.

---

## 6. `lit_module.py` vs `lit_module_old.py`

[`aria_nbv/lightning/lit_module_old.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/lightning/lit_module_old.py) (32 KB) sits next to [`lit_module.py`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/lightning/lit_module.py) (45 KB). It is dead code. Delete it.

---

## 7. `app/state_types.py` — Multiple Small Cache Dataclasses That Duplicate a Pattern

[`DataCache`, `CandidatesCache`, `DepthCache`, `PointCloudCache`, `RriCache`](https://github.com/JanDuchscherer104/ARIA-NBV/blob/main/aria_nbv/aria_nbv/app/state_types.py#L65-L95) are five structurally identical `@dataclass(slots=True)` session-cache containers, each with `cfg_sig: str | None = None` and a single payload field. This is a repeated pattern that could be a single generic `SessionCache[T]` — or at minimum should not all be bespoke dataclasses.

---

## Summary: What Needs to Happen

1. **Delete `aria_nbv/data/` files that are now duplicated in `data_handling/`** — or make `data/` thin re-export shims pointing to `data_handling/`. Do not maintain both in parallel with live logic in each.
2. **All `_pad_*`, `_stack_*`, collation helpers on `VinOracleBatch`** must exist in exactly one file.
3. **`_to_numpy`, `_histogram_overlay`, `_plot_slice_grid`, `_plot_hist_counts_mpl`** must live only in `utils/plotting.py`; all other `plotting.py` modules must import them.
4. **`_resolve_cache_dir` validator** must be a shared utility, not copy-pasted into every config class.
5. **`lit_module_old.py`** must be deleted.
6. **`vin_snippet_utils.py`** helper functions (`build_vin_snippet_view`, hash function, empty-snippet stub) should move to `utils/` and be imported from there.
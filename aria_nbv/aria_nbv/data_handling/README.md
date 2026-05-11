# Data Handling

`aria_nbv.data_handling` owns the typed boundary between upstream ASE/ATEK/EFM
assets and the training/evaluation objects consumed by ARIA-NBV. The package
root is the public API; private modules are implementation details unless they
are explicitly exported from `__init__.py`.

The central contract is actor/oracle separation. Actor-visible data comes from
observed snippets, MPS/EVL evidence, predicted or tracked OBBs, candidate
poses, and logged history. ASE meshes, GT OBBs, target crops, rendered oracle
depths, and RRI labels are supervision/evaluation assets. Invalid samples,
targets, or candidates are represented with masks and reason codes, never as
low RRI values.

## Current Public Surface

- Raw snippet access: `AseEfmDatasetConfig`, `AseEfmDataset`,
  `EfmSnippetView`, `VinSnippetView`, and snippet-loader helpers.
- One-step VIN/oracle batches: `VinOracleBatch`,
  `VinOracleOnlineDatasetConfig`, `VinDatasetSourceConfig`, and
  `VinOfflineSourceConfig`.
- Immutable one-step cache: `VinOfflineWriterConfig`, `VinOfflineWriter`,
  `VinOfflineDatasetConfig`, `VinOfflineStoreConfig`, `VinOfflineManifest`,
  and `VinOfflineIndexRecord`.
- Target and rollout replay: `ActorVisibleTargetSelector`,
  `TargetSelectorConfig`, `RolloutDatasetWriterConfig`,
  `RolloutDatasetWriter`, `RolloutZarrStoreConfig`,
  `RolloutZarrStoreReader`, and `validate_rollout_zarr_store`.
- Diagnostics: `collect_vin_offline_dataset_stats`,
  `collect_vin_offline_dataset_coverage`, and
  `collect_offline_visual_inventory`.

The removed oracle-cache, VIN-snippet-cache, compatibility wrapper, and legacy
migration modules must not be reintroduced. Runtime imports should use root
package exports, or private modules only from inside this package.

## Architecture Direction

The target architecture is logical unification over physically separated
stores. `vin_offline` remains immutable and stores expensive one-step substrate
outputs. Multi-step target-conditioned samples live in a sharded rollout
sidecar that references VIN rows and raw ASE assets by stable lineage. A joined
reader gives training and inspection a single API without copying backbone
tensors, full meshes, or raw camera streams into every rollout sample.

```mermaid
flowchart LR
  Raw["ASE / ATEK / EFM assets<br/>RGB, calibration, trajectory, semidense points, meshes"]
  VinWriter["VinOfflineWriter<br/>one-step candidate/oracle/cache build"]
  VinStore["vin_offline<br/>strict immutable indexed shards"]
  VinDataset["VinOfflineDataset<br/>sample or batch reader"]
  TargetSel["ActorVisibleTargetSelector<br/>V1 targets from observed records"]
  CandidateGen["Candidate mixture generator<br/>finite cal(Q)_t with provenance"]
  Oracle["Target RRI scorer<br/>GT-only crop and labels"]
  RolloutWriter["RolloutDatasetWriter<br/>policies, chains, masks"]
  RolloutStore["rollouts_v1<br/>sharded multi-step replay collection"]
  Joined["RolloutJoinedDataset<br/>training and inspection API"]
  Inspect["Rerun / Streamlit<br/>store summaries and selected traces"]

  Raw --> VinWriter --> VinStore
  VinStore --> VinDataset
  Raw --> VinDataset
  VinDataset --> TargetSel --> CandidateGen
  VinDataset --> Oracle
  CandidateGen --> RolloutWriter
  Oracle --> RolloutWriter
  RolloutWriter --> RolloutStore
  VinDataset --> Joined
  RolloutStore --> Joined
  RolloutStore --> Inspect
  VinDataset --> Inspect
```

The current implementation has a single standalone `rollouts.zarr` writer. The
target state is the same contract generalized into a sharded `rollouts_v1`
collection with collection-level manifests, per-shard validation, source-row
audit tables, and configurable depth-retention profiles.

## Interface Model

The most important interfaces are deliberately narrow: writers materialize
stores, readers validate and expose arrays, and joined datasets compose source
and rollout artifacts without mutating either one.

```mermaid
classDiagram
  class EfmSnippetView {
    +camera_streams
    +calibration
    +trajectory
    +semidense_points
    +optional_mesh
  }

  class VinOfflineWriter {
    +write()
  }

  class VinOfflineDataset {
    +__getitem__(index)
    +as_sample()
    +as_batch()
  }

  class VinOfflineSample {
    +sample_key
    +scene_id
    +snippet_id
    +backbone_blocks
    +detected_obbs
    +gt_obbs
    +optional_snippet
  }

  class ActorVisibleTargetSelector {
    +select(sample)
  }

  class RolloutDatasetWriter {
    +run()
  }

  class RolloutZarrStoreWriter {
    +write(traces)
  }

  class RolloutZarrStoreReader {
    +array(path)
    +validate()
  }

  class RolloutCollectionReader {
    +shards(split)
    +audit()
    +manifest()
  }

  class RolloutJoinedDataset {
    +__getitem__(index)
    +source_sample()
    +rollout_view()
  }

  EfmSnippetView --> VinOfflineWriter
  VinOfflineWriter --> VinOfflineDataset
  VinOfflineDataset --> VinOfflineSample
  VinOfflineSample --> ActorVisibleTargetSelector
  ActorVisibleTargetSelector --> RolloutDatasetWriter
  RolloutDatasetWriter --> RolloutZarrStoreWriter
  RolloutZarrStoreWriter --> RolloutZarrStoreReader
  RolloutCollectionReader --> RolloutZarrStoreReader
  RolloutJoinedDataset --> VinOfflineDataset
  RolloutJoinedDataset --> RolloutCollectionReader
```

`RolloutCollectionReader` and `RolloutJoinedDataset` are target-state
interfaces. The current implemented reader opens one rollout Zarr store at a
time.

## Immutable VIN Offline Store

The canonical one-step offline format is a strict indexed-shard store:

```text
vin_offline/
  manifest.json
  sample_index.jsonl
  splits/
    train.npy
    val.npy
    all.npy
  shards/
    shard-000000/
      numeric_blocks.zarr/
      records.msgpack
    shard-000001/
      numeric_blocks.zarr/
      records.msgpack
```

- `manifest.json` records store version, source configuration, materialized
  block flags, aggregate stats, and shard descriptors.
- `sample_index.jsonl` maps global sample indices to scene/snippet IDs, split
  membership, shard IDs, and shard-local rows.
- `splits/*.npy` stores deterministic train/val/all membership arrays.
- `shards/shard-XXXXXX/` stores fixed numeric blocks as Zarr arrays and
  optional diagnostic payloads as indexed MessagePack record blobs.

`OFFLINE_DATASET_VERSION` is the runtime compatibility gate. When the format
changes, bump the version and rebuild stores with `VinOfflineWriter`; readers
should fail fast on older manifests.

By default `VinOfflineStoreConfig.store_dir` resolves to
`PathConfig().offline_cache_dir / "vin_offline"`. Relative store names such as
`"vin_offline"` are resolved under `offline_cache_dir`.

Build immutable stores through the writer CLI, not legacy cache commands:

```sh
cd aria_nbv
uv run nbv-build-offline --config-path ../.configs/build_vin_offline_81286.toml
```

Use `--dry-run` to validate a writer TOML and inspect the resolved store path
without loading snippets, EVL, or writing shards.

## Target Rollout Collection

The target multi-step store is a collection of independently valid shards. It
is optimized for clean joins, deterministic sharding, resume-safe generation,
and ML-friendly scans.

```text
rollouts_v1/
  manifest.json
  dictionaries.json
  splits/
    train.json
    val.json
    test.json
  audit/
    source_rows.jsonl
    targets.jsonl
    build_summary.json
  shards/
    split=train/
      shard=000000.zarr/
        metadata/
        lineage/
        source_rows/
        mesh_refs/
        targets/
        rollouts/
        steps/
        candidates/
        q_h/
        depths/
          selected_action/
          candidate_valid/        # optional heavier profile
        diagnostics/
      shard=000001.zarr/
    split=val/
      shard=000000.zarr/
```

Shards should be assigned by split plus bounded VIN source-row chunks. This
keeps shard sizes predictable, avoids sample-level leakage across final splits,
and gives Slurm/DSS jobs a simple resume key. Scene-level split boundaries are
still owned by the source split manifest; a shard must not mix train/val/test.

### Store Responsibilities

| Group          | Responsibility                                                      | Redundancy rule                                                            |
| -------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `metadata/`    | Schema, field-retention profile, depth profile, build stats.        | One copy per shard.                                                        |
| `lineage/`     | Source manifest hashes, config hashes, split hash, policy ids.      | Required for every non-synthetic shard.                                    |
| `source_rows/` | VIN sample keys, source row ids, scene/snippet ids.                 | References source rows; does not copy VIN tensors.                         |
| `mesh_refs/`   | Full and simplified mesh URI/hash/version records.                  | Full meshes remain external.                                               |
| `targets/`     | Actor-visible selected targets plus GT-match evaluation fields.     | Target crops may be embedded once per target.                              |
| `rollouts/`    | One row per rollout chain and policy recipe.                        | No candidate arrays duplicated here.                                       |
| `steps/`       | One row per time step in a rollout chain.                           | Links selected candidate by row id.                                        |
| `candidates/`  | Full-shell candidate rows, masks, provenance, labels.               | Invalid candidates stay as masked rows.                                    |
| `q_h/`         | Padded candidate-query tensors for finite-candidate value learning. | Derived view over `steps` and `candidates`.                                |
| `depths/`      | ML-ready counterfactual depth renders.                              | Store selected-action depths by default; all-valid depths only by profile. |
| `diagnostics/` | Optional summaries and retained heavy debug payloads.               | Never required for training.                                               |

Invalid source rows or invalid target attempts should be persisted in
`audit/*.jsonl`, not fabricated as target-RRI samples. Candidate-level
invalidity belongs inside `candidates/` because it is part of the finite action
set and future invalidity learning.

## Individual Multi-Step Sample

One trainable multi-step sample is a joined view over one source row, one
selected target, one rollout chain, its step rows, and the candidate tables at
those steps:

```text
multi_step_sample/
  source/
    sample_key
    scene_id
    snippet_id
    source_row_id
    vin_offline_manifest_hash
    cached_backbone_ref
    raw_snippet_ref
    mesh_ref
  target/
    target_row_id
    actor_visible_descriptor
    observed_obb_world
    support_summary
    gt_match_status
    gt_match_score
    target_valid_mask
    target_invalid_reason_bitset
  rollout/
    rollout_row_id
    policy_id
    horizon
    branch_factor
    beam_width
    random_seed
    final_cumulative_target_rri
    final_cumulative_scene_rri
  steps/
    step_000/
      step_row_id
      selected_candidate_row_id
      cumulative_target_rri
      candidates/
        candidate_row_id
        pose_world_cam
        pose_relative_root
        strategy_id
        mixture_id
        sampler_probability
        actor_action_mask
        oracle_label_mask
        q_train_mask
        invalid_reason_bitset
        target_rri
        scene_rri
      selected_action_depth/
        depth_m_f16
        depth_valid_mask
        znear
        zfar
        normalization
    step_001/
      ...
  q_h/
    candidate_row_id
    valid_action_mask
    q_train_mask
    one_step_target_rri
    q_target_target_rri
    bootstrap_next_step_row_id
    terminal_mask
```

In thesis notation, `cal(Q)_t` is the finite candidate set at step `t`, and
`q_(t,i)` is one candidate pose/view. The default required depth modality is
the selected-action depth render for `q_(t,i*)`, because that render is the
counterfactual observation used to advance rollout state. A heavier profile may
also store depth renders for every actor-valid `q_(t,i)` in materialized
`cal(Q)_t`.

```mermaid
flowchart TD
  S["source row<br/>VIN sample key + cached refs"]
  T["target row e<br/>actor-visible descriptor + GT label metadata"]
  R["rollout chain<br/>policy, horizon, seed"]
  Step0["step t=0<br/>state/history/budget"]
  Q0["candidate set cal(Q)_0<br/>full-shell rows + masks"]
  Pick0["selected valid q_(0,i*)"]
  Depth0["selected-action depth<br/>float16 meters + valid mask"]
  Step1["step t=1<br/>updated geometry/history"]
  Q1["candidate set cal(Q)_1"]
  Pick1["selected valid q_(1,i*)"]
  Depth1["selected-action depth"]
  Train["Q_H training view<br/>masked candidates + returns"]

  S --> T --> R --> Step0 --> Q0 --> Pick0 --> Depth0 --> Step1
  Step1 --> Q1 --> Pick1 --> Depth1 --> Train
  Q0 --> Train
  Q1 --> Train
```

## Depth Retention Profiles

Depth storage should be explicit because it dominates rollout-store size.

| Profile                         | Stored depths                                                             | Intended use                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `selected_action_depth`         | One ML-ready depth map per rollout step for the selected valid candidate. | Required default; advances counterfactual state and supports Rerun inspection.                   |
| `selected_action_plus_retained` | Selected-action depths plus retained oracle-lookahead beam actions.       | Debugging headroom and beam-chain evidence.                                                      |
| `all_valid_candidate_depth`     | Depth map for every actor-valid candidate row in materialized `cal(Q)_t`. | Optional heavier profile for visual candidate-token learning or dense candidate-depth ablations. |

Depth maps should be stored as compressed metric `float16` arrays with a
separate boolean valid mask. Shard metadata must record resolution, renderer,
`znear`, `zfar`, invalid-fill policy, and normalization used by ML readers.
Resolution is fixed per shard/profile; mixed depth shapes are a validation
error. Full RGB, source depth streams, full meshes, and backbone tensors stay
in their source stores and are referenced by path/hash/version.

## Training Source

One-step Lightning training consumes VIN offline data through:

```toml
[datamodule_config.source]
kind = "offline"
train_split = "train"
val_split = "val"

[datamodule_config.source.offline]
load_backbone = true
map_location = "cpu"

[datamodule_config.source.offline.store]
store_dir = "vin_offline"
```

`VinOfflineSourceConfig` returns `VinOracleBatch` samples and disables
diagnostic record loading for the training path. Use `VinOfflineDatasetConfig`
directly when tests or diagnostics need the richer `return_format = "sample"`
path.

The target multi-step training path should consume rollout replay through a
joined reader:

```text
RolloutCollectionReader(split="train")
  + VinOfflineDataset(return_format="sample", load_backbone=true)
  -> RolloutJoinedDataset
  -> Q_H batch with source refs, target descriptor, candidate rows, masks,
     selected-action depths, and bounded target-RRI returns
```

This reader is the logical migration target. It avoids recomputing EFM/backbone
outputs while still generating new target-conditioned candidates, selected
counterfactual depths, and target-RRI labels.

## Verification

For data-handling changes, run the tightest relevant checks:

```sh
ruff format aria_nbv/aria_nbv/data_handling/<file>.py
ruff check aria_nbv/aria_nbv/data_handling/<file>.py
uv run pytest tests/data_handling/test_vin_offline_store.py
uv run pytest tests/data_handling/test_public_api_contract.py
```

For rollout-store work, include the rollout and target-selection tests:

```sh
uv run pytest tests/data_handling/test_target_selection.py
uv run pytest tests/data_handling/test_rollout_zarr_store.py
uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run
```

Broaden to Lightning datamodule tests when source selection or
training-facing batch assembly changes. Broaden to Rerun/Streamlit tests when
inspection flows or retained diagnostics change.

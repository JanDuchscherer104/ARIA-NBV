---
id: 2026-03-30_base_config_usage_audit
date: 2026-03-30
title: "BaseConfig usage audit across package"
status: done
topics: [config, utils, package-audit]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/lightning/lit_module_old.py
  - aria_nbv/aria_nbv/lightning/lit_datamodule.py
  - aria_nbv/aria_nbv/lightning/lit_module.py
  - aria_nbv/aria_nbv/lightning/lit_trainer_callbacks.py
  - aria_nbv/aria_nbv/lightning/optimizers.py
  - aria_nbv/aria_nbv/data/downloader.py
  - aria_nbv/aria_nbv/data/efm_dataset.py
  - aria_nbv/aria_nbv/data/offline_cache.py
  - aria_nbv/aria_nbv/data/vin_oracle_datasets.py
  - aria_nbv/aria_nbv/data/vin_snippet_cache.py
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/data_handling/efm_dataset.py
  - aria_nbv/aria_nbv/data_handling/oracle_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_datasets.py
  - aria_nbv/aria_nbv/rendering/candidate_depth_renderer.py
  - aria_nbv/aria_nbv/rendering/efm3d_depth_renderer.py
  - aria_nbv/aria_nbv/rendering/pytorch3d_depth_renderer.py
  - aria_nbv/aria_nbv/rri_metrics/logging.py
  - aria_nbv/aria_nbv/rri_metrics/oracle_rri.py
  - aria_nbv/aria_nbv/vin/backbone_evl.py
  - aria_nbv/aria_nbv/vin/model_v3.py
  - aria_nbv/aria_nbv/vin/pose_encoders.py
  - aria_nbv/aria_nbv/vin/pose_encoding.py
  - aria_nbv/aria_nbv/vin/traj_encoder.py
  - aria_nbv/aria_nbv/vin/experimental/model.py
  - aria_nbv/aria_nbv/vin/experimental/model_v1_SH.py
  - aria_nbv/aria_nbv/vin/experimental/model_v2.py
  - aria_nbv/aria_nbv/vin/experimental/pointnext_encoder.py
  - aria_nbv/aria_nbv/vin/experimental/pose_encoders.py
  - aria_nbv/aria_nbv/vin/experimental/pose_encoding.py
  - aria_nbv/aria_nbv/vin/experimental/spherical_encoding.py
  - aria_nbv/aria_nbv/app/config.py
  - aria_nbv/aria_nbv/interpretability/attribution.py
  - aria_nbv/aria_nbv/pipelines/oracle_rri_labeler.py
  - aria_nbv/aria_nbv/pose_generation/candidate_generation.py
---

task
- Audit every `BaseConfig` subclass under `aria_nbv/aria_nbv` and align package-local configs with the non-generic `target` interface.

method
- Replaced every package-local `class X(BaseConfig[...])` declaration with `class X(BaseConfig)`.
- Re-scanned the package for remaining `BaseConfig[` and `NoTarget` usage.
- Confirmed the remaining subclasses without `target` or `setup_target` are intentional container-style configs rather than factories.

verification
- `ruff format $(git diff --name-only -- aria_nbv/aria_nbv | rg '\.py$')`
- `ruff check --ignore N999 $(git diff --name-only -- aria_nbv/aria_nbv | rg '\.py$')`
- `aria_nbv/.venv/bin/python -m py_compile $(git diff --name-only -- aria_nbv/aria_nbv | rg '\.py$')`
- `aria_nbv/.venv/bin/python -m pytest aria_nbv/tests/test_base_config_toml.py`

canonical state impact
- None.

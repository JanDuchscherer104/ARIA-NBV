# Config Ownership

`aria_nbv.configs` owns typed configuration for durable package workflows:
path resolution, W&B integration, Optuna search, and related factory inputs.

Configs should stay portable and construct runtime objects through explicit
factory methods rather than relying on raw dictionaries or host-local defaults.

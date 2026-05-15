"""Target-conditioned rollout generation and replay Zarr stores.

`aria_nbv.rollouts` owns the multi-step thesis replay surface. It composes
VIN offline samples, actor-visible target selection, finite candidate
generation, and target-RRI oracle scoring into standalone rollout artifacts.
Raw snippet access and immutable VIN offline stores remain in
`aria_nbv.data_handling`.
"""

from .dataset_writer import (
    RolloutDatasetWriter,
    RolloutDatasetWriterConfig,
    RolloutDatasetWriterStats,
    RolloutRecipeConfig,
)
from .manifest import (
    ROLLOUT_MANIFEST_FILENAME,
    ROLLOUT_MANIFEST_VERSION,
    RolloutStoreInvocation,
    RolloutStoreManifestContext,
)
from .trace import (
    INVALID_REASON_CODES,
    INVALID_REASON_VERSION,
    RolloutLineage,
    RolloutZarrRecord,
)
from .zarr_store import (
    DEFAULT_RETURN_SEMANTICS,
    ROLLOUT_ZARR_SCHEMA_ID,
    ROLLOUT_ZARR_SCHEMA_VERSION,
    RolloutZarrStoreConfig,
    RolloutZarrStoreReader,
    RolloutZarrValidationResult,
    RolloutZarrWriteResult,
    validate_rollout_zarr_store,
    write_rollout_zarr_store,
)

__all__ = [
    "DEFAULT_RETURN_SEMANTICS",
    "INVALID_REASON_CODES",
    "INVALID_REASON_VERSION",
    "ROLLOUT_ZARR_SCHEMA_ID",
    "ROLLOUT_ZARR_SCHEMA_VERSION",
    "ROLLOUT_MANIFEST_FILENAME",
    "ROLLOUT_MANIFEST_VERSION",
    "RolloutDatasetWriter",
    "RolloutDatasetWriterConfig",
    "RolloutDatasetWriterStats",
    "RolloutLineage",
    "RolloutRecipeConfig",
    "RolloutStoreInvocation",
    "RolloutStoreManifestContext",
    "RolloutZarrStoreConfig",
    "RolloutZarrStoreReader",
    "RolloutZarrRecord",
    "RolloutZarrValidationResult",
    "RolloutZarrWriteResult",
    "validate_rollout_zarr_store",
    "write_rollout_zarr_store",
]

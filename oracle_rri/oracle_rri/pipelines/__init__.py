"""Reusable pipelines built on top of the oracle RRI components.

The dashboard (Streamlit) is intentionally kept as a thin visualization layer.
Training and offline label generation should use the non-UI pipelines in this
package so data-flow, batching, and performance controls remain explicit.
"""

from .oracle_rri_labeler import OracleRriLabeler, OracleRriLabelerConfig, OracleRriSample

__all__ = [
    "OracleRriSample",
    "OracleRriLabeler",
    "OracleRriLabelerConfig",
]

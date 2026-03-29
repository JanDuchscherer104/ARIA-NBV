# ruff: noqa: INP001
"""Bootstrap OpenPoints import path on interpreter startup."""

import openpoints_shim

openpoints_shim.bootstrap()

# `aria_nbv.data`

`aria_nbv.data` is the legacy compatibility surface for code that has not yet
switched to the canonical data package.

The active owner for raw ASE/EFM snippets, oracle caches, VIN caches, and
shared training-data contracts is [`aria_nbv.data_handling`](../data_handling/README.md).

Legacy modules in this package are compatibility aliases that bind old import
paths such as `aria_nbv.data.offline_cache` to their canonical owners under
`aria_nbv.data_handling`. New code should import from `aria_nbv.data_handling`
directly.

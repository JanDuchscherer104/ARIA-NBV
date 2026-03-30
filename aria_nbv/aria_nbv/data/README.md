# `aria_nbv.data`

`aria_nbv.data` is deprecated.

The canonical owner for raw ASE/EFM snippets, oracle caches, VIN caches, cache
coverage utilities, and shared training-data contracts is
[`aria_nbv.data_handling`](../data_handling/README.md).

This package now retains only residual utilities that have not yet moved.
Mirrored compatibility modules such as `aria_nbv.data.efm_views`,
`aria_nbv.data.offline_cache`, and `aria_nbv.data.plotting` were removed; new
code should import from `aria_nbv.data_handling` or `aria_nbv.utils` directly.

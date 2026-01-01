# VIN v2 proposals embedded in docs landing page

## Goal

Make the main Quarto landing page (`docs/index.qmd`) contain the VIN v2 feature/encoding checklist so it’s visible without navigating to a separate page.

## Change

- Embedded the full content of `docs/contents/impl/vin_v2_feature_proposals.qmd` into `docs/index.qmd` under a new top-level section.
- Adjusted internal links in the embedded copy to point to the correct site paths from the docs root (e.g. `contents/impl/vin_nbv.qmd` instead of `vin_nbv.qmd`).

## Verification

- `cd docs && quarto render index.qmd --to html` succeeds.


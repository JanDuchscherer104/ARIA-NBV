# CORAL Implementation Delta Summary (2026-01-29)

## Scope
- Reviewed `oracle_rri/oracle_rri/rri_metrics/coral.py` vs coral-pytorch reference.
- Updated paper + slides to document deltas and cite coral-pytorch docs.
- Added monotone bin value equation to shared macros.

## Key Deltas (ours vs coral-pytorch)
- Wrapper converts ordinal labels to CORAL levels internally.
- Converts cumulative probabilities to class marginals and expected RRI.
- Learnable monotone bin representatives (`u_k`) via softplus deltas.
- Optional bias initialization from fitted class priors.
- Balanced BCE and focal loss variants for threshold imbalance.
- Diagnostics: monotonicity violation rate + relative-to-random baseline.

## Files Updated
- `oracle_rri/oracle_rri/rri_metrics/coral.py` (reference)
- `docs/typst/shared/macros.typ` (added `eqs.coral.bin_values`)
- `docs/typst/paper/sections/07-training-objective.typ` (new deltas subsection)
- `docs/typst/slides/slides_4.typ` (new delta slide)

## Notes
- Citation used: `@coral-pytorch-2025` in `docs/references.bib`.

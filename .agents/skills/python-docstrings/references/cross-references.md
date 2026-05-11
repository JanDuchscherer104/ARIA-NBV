# Cross-References

Use Quarto-compatible Markdown for internal symbols and external sources.
Quartodoc currently preserves many Sphinx/RST roles as literal text in generated
`.qmd` pages, so use Markdown links, backticks, and Markdown math unless a local
render proves that role conversion works.

## Preferred Internal References

- Use backticks for symbol names: `VinOfflineDataset`, `OracleRRI.score`,
  `aria_nbv.data_handling`, `OFFLINE_DATASET_VERSION`.
- Use Markdown links only when the target is stable and helpful:
  `[RRI theory](../../contents/theory/rri_theory.qmd)`.
- Prefer module/class docstrings for dense cross-surface contracts; avoid
  repeating the same link on every method.

Examples:

- ``Return a `RunPlan` built from the validated request.``
- ``Translate updates through `translate_slam_update`.``
- ``See `RunService.start_run` for the entrypoint.``
- ``Keep offline-store semantics in `aria_nbv.data_handling`.``
- ``The payload is stored in `RunSnapshot.slam`.``
- ``Respect `DEFAULT_MAX_FRAMES_IN_FLIGHT` when tuning credits.``

## Math

Use Markdown math:

- Inline: `$P_t \cup P_q$`
- Display:

```text
$$
\mathrm{RRI}(q)=\frac{D(P_t,M)-D(P_t\cup P_q,M)}{D(P_t,M)+\epsilon}.
$$
```

Use raw docstrings when LaTeX backslashes appear:

```python
r"""Compute improvement $D(\mathcal{P}_t,M)-D(\mathcal{P}_{t\cup q},M)$."""
```

## External References

Use markdown links for external material:

- API docs
- research papers
- tutorials
- conceptual references

Examples:

- `See [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).`
- `The anomaly score follows the [Isolation Forest paper](https://seppe.net/aa/papers/iforest.pdf).`

Do not use Sphinx roles for external targets. Keep external links selective and
relevant.

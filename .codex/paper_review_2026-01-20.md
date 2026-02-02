# Paper review (Typst): correctness + consistency issues

Date: 2026-01-20
Scope: `docs/typst/paper/main.typ` and included `sections/*.typ`.

## Critical correctness / scope mismatches

- `docs/typst/paper/sections/09-diagnostics.typ` currently claims training-based findings:
  - “Early experiments reveal … the model can collapse …” and “We log … per epoch …” read like *completed* learning experiments, but the project status is currently “oracle-label computation + diagnostics; learned policy is future work”.
  - Action: rephrase as *expected* failure modes / planned diagnostics, or move this section to “Future work” / appendix.

- `docs/typst/paper/sections/10-discussion.typ` uses present-tense “mitigate” language for modules that are not yet demonstrated via results:
  - “Semidense projection features and frustum attention mitigate …”
  - Action: rephrase as “in our planned architecture …” unless backed by reported experiments.

## Reproducibility / configuration inconsistencies

- Candidate radius mismatch between config table(s) and runnable config:
  - Appendix center sampling table uses `max_radius = 3.0 m` (`docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`), but the paper’s “oracle label configuration” table + runnable TOML use `max_radius = 2.9` (`docs/typst/paper/sections/08-system-pipeline.typ`, `.configs/paper_figures_oracle_labeler.toml`).
  - Action: pick one value (likely `2.9` to match existing figures) and update all tables/text accordingly.

- Backface culling is inconsistent across paper tables/config and also appears inconsistent with effective code:
  - Appendix render table claims `cull_backfaces = false` (`docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`).
  - Runnable TOML sets `cull_backfaces = true` (`.configs/paper_figures_oracle_labeler.toml`).
  - Code path: `oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py` hardcodes `RasterizationSettings(cull_backfaces=False)` (config field not applied).
  - Action: decide the intended behavior, then (1) make code honor the config, and (2) align paper tables + TOML to the effective behavior.

- Dataset preprocessing version likely outdated / ambiguous:
  - Paper claims “ATEK WebDataset preprocessing pipeline (v0.1.1)” (`docs/typst/paper/sections/04-dataset.typ`), but the repo pins `projectaria-atek==1.0.0` in `oracle_rri/pyproject.toml`.
  - Action: either remove the specific version (if unknown), or document the true version/source of the exported shards.

## Figure provenance / citation gaps

Some figures look sourced from external papers/docs but captions do not cite them:

- ASE modalities figure: `docs/typst/paper/sections/04-dataset.typ` uses `/figures/scene-script/ase_modalities.jpg` with no citation in caption.
- VIN-NBV diagram: `docs/typst/paper/sections/06-architecture.typ` uses `/figures/VIN-NBV_diagram.png` with no citation.
- ATEK overview: `docs/typst/paper/sections/09a-evaluation.typ` uses `/figures/atek/overview.png`; caption has no citation (even if surrounding text cites ATEK).

Action: add explicit citations/credits in figure captions (or state “recreated by us” if that’s the case).

## Clarity / notation issues to consider

- `P_t ∪ P_q` is written as a set union; implementation is closer to concatenation (multiset union). Consider clarifying in text.
- The paper uses a sequential “step t” formulation; the implemented oracle pipeline operates per snippet (often with `reference_frame_index=None`, i.e. final pose). Consider making the mapping from “t” to “snippet reference frame” explicit.


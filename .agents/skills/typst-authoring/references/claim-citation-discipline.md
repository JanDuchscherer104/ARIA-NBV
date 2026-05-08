# Claim And Citation Discipline

## Claim Taxonomy

Every sentence that does scientific work should be classifiable as one of:

1. Definition: introduces a term, symbol, metric, or scope.
2. Literature claim: summarizes prior work and requires citation.
3. Implementation fact: describes code, data flow, or repository behavior.
4. Design decision: explains an architecture, representation, metric, or
   workflow choice.
5. Empirical result: reports measured performance, runtime, ablation outcome,
   or qualitative finding.
6. Limitation: bounds what can be claimed.
7. Hypothesis / future work: explicitly marked as not yet established.

If a claim cannot be classified, it is probably filler.

## Citation Rules

- Never invent citations or bibliography keys.
- Prefer primary sources for methods, datasets, and benchmarks.
- Use review papers only for broad context.
- Avoid citation clusters that do not say what each source contributes.
- Use `[CITATION NEEDED: expected source type]` only as a temporary marker.
- Do not use a citation as a substitute for explaining the connection to
  ARIA-NBV.

## Evidence Gate

For each non-obvious claim, verify at least one evidence path:

- `@bib_key` in `docs/references.bib`;
- code path or generated context artifact;
- table/figure/result in `docs/typst/shared/data`;
- explicit limitation or hypothesis wording.

For advisor-facing scientific claims, run:

```bash
make kg-claim-check KG_CLAIM='...'
```

Do not require `kg-claim-check` for skill-only edits or purely mechanical
Typst fixes.

## Hedging

Use strong verbs only when the evidence is strong:

- demonstrates / shows: direct empirical evidence;
- suggests / indicates: indirect evidence or limited samples;
- may / could: hypotheses and future work.

Do not stack hedges.

## Results Prose Pattern

Use claim-first paragraphs:

```text
The VIN proxy ranks oracle-preferred candidates more reliably after adding semi-dense projection features. In the offline validation split, ... . This improvement is consistent with the feature design, because ... .
```

Avoid procedure-first prose:

```text
We trained the model and then evaluated the predictions. Figure X shows the results.
```

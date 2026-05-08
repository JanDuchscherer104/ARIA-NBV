# ARIA-NBV Notation Policy

## Source Of Truth

Use the shared Typst library in `docs/typst/shared`:

```text
docs/typst/shared/
  symbols.typ       # facade: symb.frame, symb.vin, symb.oracle, ...
  equations.typ     # facade: eqs.rri, eqs.vin, eqs.entity, ...
  math.typ          # helpers such as T(A, B)
  terms.typ         # generated glossary term facade
  glossary.typ      # glossary source
  macros.typ        # reusable document macros
```

Import relative to the file being edited. From a thesis proposal section under
`docs/typst/thesis/sections/proposal/`, the shared library is reached through
`../../../shared/...`; fixtures under `.agents/skills` can use root-relative
imports with `--root .`.

## Shared Modules

Use these facades before writing raw notation:

```typst
#symb.frame.w
#symb.frame.r
#symb.frame.cq
#symb.vin.pose_emb
#symb.vin.sem_proj
#symb.vin.field_v
#symb.oracle.candidates
#symb.oracle.rri
#symb.entity.rri_total
#eqs.rri.rri
#eqs.entity.objective
#T(symb.frame.w, symb.frame.r)
```

## Typed Convention

Use a typed convention, not blanket bolding:

- Bold data-bearing vectors, matrices, tensors, fields, embeddings,
  point-cloud collections, image/depth tensors, voxel tensors, and learned
  feature bundles.
- Usually leave scalar losses, scalar metrics, abstract sets, operators,
  indices, dimensions, and weights unbolded unless the shared symbol says
  otherwise.
- Use `cal(...)` or the shared symbol for mathematical sets and collections.
- Use `op("...")` for named mathematical operators.

Examples:

```typst
$ bold(x)_q, bold(E)_q, bold(F)_v, bold(V)_"occ"^"pr" $
$ cal(Q), cal(E), op("argmax")_(q in cal(Q)) $
```

## Adding Missing Notation

When a new symbol is needed:

1. Pick the domain module: `symbols/vin.typ`, `symbols/oracle.typ`,
   `symbols/entity.typ`, and so on.
2. Add a comment describing semantics and usage.
3. Export through `symbols.typ` if a new module is added.
4. Add reusable formulae under `equations/*.typ` and export through
   `equations.typ`.
5. Use the shared symbol/equation in the document.
6. Compile a fixture or the affected document.

Do not create a local one-off alias in a thesis section if the symbol will
recur.

## ARIA-NBV Prose Convention

Use project terms consistently:

- ARIA-NBV, not ARIA NBV.
- Next-best view (NBV) on first definition, NBV afterwards.
- Relative Reconstruction Improvement (RRI) on first definition, RRI
  afterwards.
- Semi-dense point cloud, candidate view, candidate pose, target-specific RRI,
  VIN proxy, oracle labeler.

Avoid uncontrolled synonyms in thesis prose.

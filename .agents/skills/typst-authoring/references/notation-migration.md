# Shared-Notation Migration Notes

Use this when a proposal/thesis edit touches notation that appears in more
than one section. The May 2026 shared-notation pass locked the core convention:
abstract geometry uses `cal(...)`, tensors/features use `bold(...)`, candidate
sets use `cal(Q)_t`, abstract states use plain `s`, and thesis-core RRI uses
point-mesh error `D` rather than generic `CD`.

## Locked Migration Targets

These patterns should not reappear in real proposal/thesis Typst sources:

| Old pattern | Use instead |
| --- | --- |
| `bold(cal(P))`, `bold(cal(Q))`, `bold(cal(M))`, `bold(cal(F))` | `cal(P)`, `cal(Q)`, `cal(M)`, `cal(F)` |
| `bold(Q)_t` as candidate table | `cal(Q)_t` or `#symb.rl.candidate_table` |
| `bold(q)_(t,i)` as candidate pose | `q_(t,i)` or `#symb.rl.candidate_qti` |
| `bold(s)_t^"obs"`, `bold(s)_t^"cf0"` | `s_t^"obs"`, `s_t^"cf0"` |
| `cal(A)_t^e + cal(C)_t^e` | `D_(P -> M,t)^e + D_(M -> P,t)^e` |
| `CD(...)` for thesis-core ARIA-NBV error | `D(...)` / `#symb.oracle.err` |
| `S O(2)` | `op("SO")(2)` |

## Shared Owners

| Concept | Owner |
| --- | --- |
| Abstract points, candidates, directional error | `docs/typst/shared/symbols/oracle.typ` |
| ASE meshes/faces/target crops | `docs/typst/shared/symbols/ase.typ` |
| Observed and counterfactual point sets | `docs/typst/shared/symbols/obs.typ` |
| Abstract states, masks, candidate tokens, `Q_H` | `docs/typst/shared/symbols/rl.typ` |
| Target error, endpoint gain, headroom, recovery | `docs/typst/shared/symbols/entity.typ` and `equations/entity.typ` |
| Point-mesh RRI equations | `docs/typst/shared/equations/rri.typ` |

## Migration Rule

Migrate advisor-facing Typst when touching affected prose or equations.
Broader Quarto/theory migration can be deferred unless a page directly
contradicts proposal/advisor notation. Keep temporary compatibility keys in the
shared API until migrated documents are stable, then remove aliases in a
separate cleanup.

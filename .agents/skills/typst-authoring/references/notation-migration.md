# Shared-Notation Migration Notes

Use this when a proposal/thesis edit touches notation that appears in more
than one section. This pass records drift; it does not require migrating the
current proposal unless the user asks for proposal edits.

## Probation Symbols

The following inline patterns recur or are expected to recur and should be
shared before the next substantial proposal/thesis notation pass:

| Inline pattern | Target owner |
| --- | --- |
| `bold(s)_t^"obs"` | `docs/typst/shared/symbols/obs.typ` or `symbols/rl.typ` |
| `bold(s)_t^"cf0"` | `docs/typst/shared/symbols/rl.typ` |
| `bold(z)_e` | `docs/typst/shared/symbols/entity.typ` |
| `Q_(H,theta)` / `Q_H` | `docs/typst/shared/equations/rl.typ` |
| `Delta_t^e` | `docs/typst/shared/equations/entity.typ` or `metrics.typ` |
| `J_e^(H)` | `docs/typst/shared/equations/entity.typ` or `metrics.typ` |
| `G_t^(H)` | `docs/typst/shared/equations/rl.typ` |
| `bold(F)_t^"EVL"` | `docs/typst/shared/symbols/vin.typ` or `obs.typ` |
| `bold(O)_t^"pred"` | `docs/typst/shared/symbols/entity.typ` or `obs.typ` |

## Known Conflict

The shared library currently exposes semi-dense points as
`bold(cal(P))^"semi"` through `symb.obs.points_semi`, while proposal prose has
used `bold(P)_t^"semi"` for a timestep-indexed state. The next notation pass
must decide whether the time-indexed state is a separate shared symbol or
whether proposal usage should adopt the collection-valued shared symbol with a
clear timestep wrapper.

## Migration Rule

Migrate only when touching the affected prose or equations for content work.
Do not mix a large notation migration into unrelated prose polish.

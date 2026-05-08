#import "/docs/typst/shared/symbols.typ": symb
#import "/docs/typst/shared/equations.typ": eqs
#import "/docs/typst/shared/math.typ": T

#set page(paper: "a4", margin: 1.2cm)
#set text(size: 10pt)
#set heading(numbering: none)

= Shared Notation Smoke Fixture

This fixture verifies that thesis authoring uses `docs/typst/shared` instead
of inventing local notation.

== Symbols

Pose embedding: #symb.vin.pose_emb

Semi-dense projection statistics: #symb.vin.sem_proj

Candidate set: #symb.oracle.candidates

Entity-aware total RRI: #symb.entity.rri_total

Frame transform helper: #T(symb.frame.w, symb.frame.r)

== Equations

#eqs.rri.rri

#eqs.entity.objective

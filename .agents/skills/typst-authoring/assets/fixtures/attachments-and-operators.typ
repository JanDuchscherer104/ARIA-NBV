#set page(paper: "a4", margin: 1.2cm)
#set text(size: 10pt)
#set heading(numbering: none)

= Typst Attachment / Operator Regression Fixture

This fixture should compile and render cleanly. Inspect the output to verify
that argument lists and output indices attach to the intended expression.

== Attachment Spacing

Bad pattern to avoid in thesis source:

```typst
$ op("IoU")_"3D"(hat(bold(B))_(hat(e)), bold(B)_e^"GT") $
```

Preferred pattern:

$ op("IoU")_"3D" (hat(bold(B))_(hat(e)), bold(B)_e^"GT") $

== Output Indexing

Preferred output indexing:

$ bold(h)_i = (op("Transformer")_theta (bold(X)_t))_i $

Preferred prediction head:

$ hat(r)_q = op("MLP")_phi (bold(h)_q) $

== Roman Labels And Bold Data Tensors

$ bold(V)_"occ"^"pr", quad bold(V)_"count"^"norm", quad bold(s)_"proj", quad bold(F)_v $

== Abstract Sets Stay Unbolded Unless Shared Symbols Define Otherwise

$ q^* = op("argmax", limits: #true)_(q in cal(Q)) "RRI"(q) $

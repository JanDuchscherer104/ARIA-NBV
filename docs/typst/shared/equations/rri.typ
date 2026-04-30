#import "../symbols.typ": symb

#let rri = (
    cd: $
      "CD"(#symb.oracle.points, #symb.ase.mesh) =
      #symb.oracle.acc (#symb.oracle.points, #symb.ase.mesh) + #symb.oracle.comp (#symb.oracle.points, #symb.ase.mesh)
    $,
    acc: $
      #symb.oracle.acc (#symb.oracle.points, #symb.ase.mesh) =
      (1)/(||#symb.oracle.points||) sum_(bold(p) in #symb.oracle.points) min_(bold(f) in #symb.ase.faces) d(bold(p), bold(f))^2
    $,
    comp: $
      #symb.oracle.comp (#symb.oracle.points, #symb.ase.mesh) =
      (1)/(||#symb.ase.faces||) sum_(bold(f) in #symb.ase.faces) min_(bold(p) in #symb.oracle.points) d(bold(p), bold(f))^2
    $,
    union: $
      #(symb.oracle.points) _(t union q) = #(symb.oracle.points) _t union #symb.oracle.points_q
    $,
    rri: $
      "RRI"(q) =
      ("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) - "CD"(#(symb.oracle.points) _t union #symb.oracle.points_q, #symb.ase.mesh))
      / ("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
    $,
    greedy: $ q_star = op("argmax", limits: #true)_(q in #symb.oracle.candidates) "RRI"(q) $,
  )

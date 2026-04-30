#import "../symbols.typ": symb
#import "../terms.typ": RRI

#let entity = (
    objective: $
      RRI_"total"(q)
      =
      sum_(e in #symb.entity.E)
      #(symb.entity.w) _e dot #(symb.oracle.rri) _e
      +
      #symb.entity.lambda_scene dot #symb.oracle.rri
    $,
  )

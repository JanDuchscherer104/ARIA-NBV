#import "../symbols.typ": symb

#let features = (
    film: $
      #(symb.vin.global) _i^("film")
      = (1 + #(symb.vin.gamma) _i) dot.op #(symb.vin.global) _i + #(symb.vin.beta) _i
    $,
    semidense_validity: $
      m_(i,j)
      =
      bb(1)["finite"] dot bb(1)[z_(i,j) > 0] dot
      bb(1)[0 <= u_(i,j) < W_i] dot bb(1)[0 <= v_(i,j) < H_i]
    $,
    semidense_visibility: $
      v_i^("sem")
      = (sum_j w_(i,j) m_(i,j)) / (sum_j w_(i,j) f_(i,j))
    $,
  )

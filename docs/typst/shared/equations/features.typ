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
    direction_unit: $
      bold(d)_k(bold(v))
      =
      (bold(c)_k - bold(v)) / norm(bold(c)_k - bold(v))_2
    $,
    direction_memory_sh: $
      #symb.vin.dir_memory (bold(v))
      =
      sum_(k < t) w_k(bold(v))
      #symb.vin.sh_basis (bold(d)_k(bold(v)))
    $,
    direction_memory_moment: $
      #symb.vin.dir_moment (bold(v))
      =
      sum_(k < t) w_k(bold(v))
      bold(d)_k(bold(v)) bold(d)_k(bold(v))^top
    $,
    direction_novelty: $
      nu_(t,i)^"dir"(bold(v))
      =
      1 -
      (bold(d)_(t,i)(bold(v))^top #symb.vin.dir_moment (bold(v)) bold(d)_(t,i)(bold(v)))
      /
      (op("tr") (#symb.vin.dir_moment (bold(v))) + epsilon)
    $,
  )

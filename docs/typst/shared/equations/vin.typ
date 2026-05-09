#import "../symbols.typ": symb

#let vin = (
    // Observation-count normalization used as a voxel-coverage proxy.
    counts_norm: $
      #symb.vin.counts_norm
      = ("log"(1 + #symb.vin.counts)) / ("log"(1 + "max"(#symb.vin.counts)))
    $,
    // Unknown mask + new-surface prior derived from counts and occupancy.
    new_surface_prior: $
      #symb.vin.unknown = 1 - #symb.vin.counts_norm,
      quad #symb.vin.new_surface_prior = #symb.vin.unknown dot.op #symb.vin.occ_pr
    $,
    // Optional auxiliary regression combined with the CORAL loss.
    loss_total: $ #symb.vin.loss = #(symb.vin.loss) _"coral" + lambda dot #(symb.vin.loss) _"reg" $,
    aux_reg_mse: $
      #(symb.vin.loss) _"reg"
      = (1)/(N) sum_i (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i)^2
    $,
    aux_reg_huber: $
      #(symb.vin.loss) _"reg"
      = (1)/(N) sum_i "Huber"_1 (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i)
    $,
    huber: $
      "Huber"_1 (e) = { 0.5 e^2 "if" |e| <= 1; |e| - 0.5 "otherwise" }
    $,
    aux_weight: $
      lambda_"reg" (t)
      = max(lambda_0 dot gamma^t, lambda_"min")
    $,
  )

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
    target_error: $
      #symb.entity.target_error
      =
      A_t^e + C_t^e
    $,
    target_rri_reward: $
      #symb.entity.target_reward
      =
      (#symb.entity.target_error - #symb.entity.target_error_next)
      /
      (#symb.entity.target_error + epsilon)
    $,
    finite_horizon_return: $
      #symb.entity.return_h
      =
      sum_(k=0)^(H - 1) gamma^k r_(t+k)^e
    $,
    endpoint_gain: $
      #symb.entity.endpoint_gain
      =
      (#symb.entity.target_error_0 - #symb.entity.target_error_H)
      /
      (#symb.entity.target_error_0 + epsilon)
    $,
    log_gain: $
      #symb.entity.log_gain
      =
      log(#symb.entity.target_error_0 + epsilon)
      -
      log(#symb.entity.target_error_H + epsilon)
    $,
    lookahead_headroom: $
      #symb.entity.lookahead_headroom
      =
      J_e^((H)) (pi_"oracle-look")
      -
      J_e^((H)) (pi_"oracle-1")
    $,
    q_recovery: $
      #symb.entity.q_recovery
      =
      (
        J_e^((H)) (pi_Q)
        -
        J_e^((H)) (pi_"learned-1")
      )
      /
      (
        J_e^((H)) (pi_"oracle-look")
        -
        J_e^((H)) (pi_"learned-1")
        +
        epsilon
      )
    $,
  )

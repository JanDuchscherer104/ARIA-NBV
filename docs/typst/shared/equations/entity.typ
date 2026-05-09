#import "../symbols.typ": symb
#import "../terms.typ": RRI

#let entity = (
    objective: $
      RRI_"total" (q)
      =
      sum_(e in #symb.entity.E)
      #(symb.entity.w) _e dot #(symb.oracle.rri) _e
      +
      #symb.entity.lambda_scene dot #symb.oracle.rri
    $,
    target_descriptor: $
      #symb.entity.target_desc
      =
      phi(
        hat(bold(B))_e,
        hat(bold(y))_e,
        hat(p)_e,
        A_e^"proj",
        n_e^"semi",
        n_e^"EVL",
        bold(T)_e^"rel"
      )
    $,
    target_match_score: $
      mu(hat(e), e)
      =
      kappa(hat(y)_(hat(e)), y_e)
      dot op("IoU")_"3D" (hat(bold(B))_(hat(e)), bold(B)_e^"GT")
      dot sigma(A_(hat(e))^"proj", n_(hat(e))^"semi", n_(hat(e))^"EVL")
    $,
    target_match_selection: $
      e^star
      =
      op("argmax", limits: #true)_(e in cal(E))
      mu(hat(e), e)
    $,
    target_match_acceptance: $
      "accept iff"
      quad
      mu(hat(e), e^star) >= tau_mu
      quad "and" quad
      mu_1 - mu_2 >= tau_"gap"
    $,
    target_error: $
      #symb.entity.target_error
      =
      #symb.entity.target_error_pm
      +
      #symb.entity.target_error_mp
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

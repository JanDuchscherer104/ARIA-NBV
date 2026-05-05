#import "../symbols.typ": symb

#let rl = (
    mdp: $
      cal(M) = (cal(S), cal(A), P, #symb.rl.r, gamma)
    $,
    nbv_mdp: $
      #symb.rl.mdp_nbv = (cal(S), cal(A), T, r_e, #symb.rl.gamma, #symb.rl.H)
    $,
    hist_ego: $
      #symb.rl.hist_ego
      =
      (
        #symb.obs.img_rgb,
        #symb.obs.pose,
        #symb.obs.points_semi,
        #(symb.vin.field_v)^"ego"
      )_(1:t)
    $,
    hist_cf: $
      #symb.rl.hist_cf
      =
      (
        #symb.rl.hist_ego,
        (
          #(symb.obs.depth)^"cf",
          #(symb.obs.vis)^"cf",
          #symb.obs.points_cf
        )_(1:t)
      )
    $,
    state_ego: $
      #(symb.rl.s) _t^"ego"
      =
      (
        #symb.rl.hist_ego,
        #(symb.rl.x) _t,
        #(symb.ase.points_semi) _t,
        #(symb.vin.field_v) _t,
        #(symb.rl.e) _t,
        #(symb.rl.b) _t
      )
    $,
    state_cf: $
      #(symb.rl.s) _t^"cf"
      =
      (
        #symb.rl.hist_cf,
        #(symb.rl.x) _t,
        #(symb.rl.m) _t,
        #(symb.rl.e) _t,
        #(symb.rl.b) _t
      )
    $,
    obs_render: $
      #(symb.rl.o) _(t+1)
      =
      cal(G)(#symb.ase.mesh, #(symb.rl.x) _(t+1))
    $,
    memory_update: $
      #(symb.rl.m) _(t+1)
      =
      cal(U)(
        #(symb.rl.m) _t,
        #(symb.rl.o) _(t+1),
        #(symb.rl.x) _(t+1)
      )
    $,
    finite_action_set: $
      #symb.rl.action_set = {#symb.oracle.candidate_qti in #symb.oracle.candidates_t : #symb.rl.validity_mask = 1}
    $,
    counterfactual_transition: $
      #(symb.oracle.points) _(t+1) = #(symb.oracle.points) _t union #(symb.oracle.points) _(q _t)
    $,
    target_rri_reward: $
      #symb.rl.reward_target = "RRI"_e(q _t mid #(symb.oracle.points) _t, #symb.ase.mesh_target)
    $,
    finite_horizon_return: $
      #symb.rl.return_h = sum_(k=0)^(#symb.rl.H - 1) #symb.rl.gamma^k r_(t+k)^e
    $,
    q_h: $
      "Q"_H(s _t, q) = bb(E)[G _t^(H) mid s _t, a _t = q]
    $,
    reward_log: $
      #(symb.rl.r) _t
      =
      "log"("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
      -
      "log"("CD"(#(symb.oracle.points) _(t+1), #symb.ase.mesh) + epsilon)
    $,
    reward_geom: $
      #(symb.rl.r) _t^"geom"
      =
      "log"("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
      -
      "log"("CD"(#(symb.oracle.points) _(t+1), #symb.ase.mesh) + epsilon)
      -
      alpha bb(1)["collision"(#(symb.rl.a) _t)]
      -
      beta c(#(symb.rl.a) _t)
    $,
    planner: $
      #(symb.rl.a) _t^star
      =
      "arg max"_(#(symb.rl.a) _(t:t+H-1))
      sum_(k=0)^(H-1) gamma^k #(symb.rl.r) _(t+k)
    $,
    q_backup: $
      y_t^Q
      =
      #(symb.rl.r) _t
      +
      gamma #(symb.rl.V) ( #(symb.rl.s) _(t+1) )
    $,
    iql_q_loss: $
      cal(L)_(#(symb.rl.Q))^"IQL"
      =
      ( #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t ) - y_t^Q )^2
    $,
    cql_loss: $
      cal(L)_(#(symb.rl.Q))^"CQL"
      =
      (1)/(2) ( #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t ) - y_t^Q )^2
      +
      alpha (
        "logsumexp"_(a in cal(A)) #(symb.rl.Q) ( #(symb.rl.s) _t, a )
        -
        #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t )
      )
    $,
    return_lambda: $
      #(symb.rl.G) _t^lambda
      =
      (1-lambda) sum_n lambda^(n-1) G_t^(n)
    $,
    leq_loss: $
      cal(L)_(#(symb.rl.V))^"LEQ"
      =
      rho_(tau)(
        #(symb.rl.V) ( #(symb.rl.s) _t ) - #(symb.rl.G) _t^lambda
      )
    $,
    gae: $
      #(symb.rl.A) _t^"GAE"
      =
      sum_(l=0)^(L-1) (gamma lambda)^l #(symb.rl.delta) _(t+l)
    $,
    ppo_clip: $
      cal(L)_(#(symb.rl.pi))^"PPO"
      =
      bb(E)[
        "min"(
          #(symb.rl.rho) _t #(symb.rl.A) _t,
          "clip"(#(symb.rl.rho) _t, 1-epsilon, 1+epsilon) #(symb.rl.A) _t
        )
      ]
    $,
    hier_policy: $
      #(symb.rl.z) _t ~ #(symb.rl.pi) _("hi")(z ; #(symb.rl.s) _t),
      quad
      #(symb.rl.a) _t ~ #(symb.rl.pi) _("lo")(a ; #(symb.rl.s) _t, #(symb.rl.z) _t)
    $,
  )

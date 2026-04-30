#let rl = (
    // RL state / observation / reward / return.
    s: $s$,
    o: $o$,
    a: $a$,
    r: $r$,
    G: $G$,
    Q: $Q$,
    V: $V$,
    pi: $pi$,
    A: $A$,
    delta: $delta$,
    rho: $rho$,
    z: $z$,
    // Pose / persistent memory / optional entity memory / budget.
    x: $bold(x)$,
    m: $bold(m)$,
    e: $bold(e)$,
    b: $b$,
    // History bundles for the two state formulations.
    hist_ego: $cal(O)^"ego"_(1:t)$,
    hist_cf: $(cal(O)^"ego", cal(O)^"cf")_(1:t)$,
  )

function yao_model(P = 2000; T = Float64, backend = nothing, kwargs...)
    # YAO: linear least-squares with k-convex constraints
    # P=2000, k=2: N = P+k = 2002 variables X(1..P+k)
    # Objective: sum_{i=1}^{P+k} (X(i) - sin(i/(P+k)))^2 / 2
    #   (GROUP TYPE SQ with SCALE 2.0: contribution = v^2 / 2.0 where v = x[i] - sin(i/(P+k)))
    # Constraints: B(i) for i=1..P (G≥0):
    #   X(i) - 2*X(i+1) + X(i+2) ≥ 0
    # Bounds: X(1) ≥ 0.08; X(P+1..P+k) = 0; all others free

    k   = 2
    NPK = P + k  # total number of variables

    # Compute RHS constants: sin(i/NPK) for i=1..NPK
    rhs_vals = T[sin(i / NPK) for i in 1:NPK]

    lvar = fill(T(-Inf), NPK)
    uvar = fill(T(Inf), NPK)
    x0   = zeros(T, NPK)

    # X(1) ≥ 0.08
    lvar[1] = T(0.08)

    # X(P+1..P+k) = 0 (fixed)
    for i in P+1:NPK
        lvar[i] = T(0); uvar[i] = T(0); x0[i] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NPK; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum_{i=1}^{NPK} (x[i] - sin(i/NPK))^2 / 2
    # SIF: S(i) has SCALE 2.0 and GROUP TYPE SQ (v^2):
    # contribution = v^2 / 2.0 where v = x[i] - rhs[i]
    ExaModels.@add_par(c, rhs_par, rhs_vals)
    ExaModels.@add_obj(c, (x[i] - rhs_par[i])^2 / T(2) for i in 1:NPK)

    # Constraints B(i) for i=1..P (G≥0): x[i] - 2*x[i+1] + x[i+2] ≥ 0
    ExaModels.@add_con(c,
        x[i] - T(2)*x[i+1] + x[i+2]
        for i in 1:P; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

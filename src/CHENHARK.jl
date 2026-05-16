# CHENHARK: Bound-constrained Linear Complementarity Problem
# N=5000 (default), NFREE=2500, NDEGEN=500
# Variables X(1..N), lvar=0 (default), start=0.5
# Objective: 0.5*Q(0)^2 + ... + 0.5*Q(N+1)^2 + L
# where Q(I) are second-difference groups (HALFL2 type)
# and L is a linear group from the q-vector of the LCP
function chenhark_model(N = 5000; NFREE = N ÷ 2, NDEGEN = N ÷ 10, T = Float64, backend = nothing, kwargs...)
    NFND = NFREE + NDEGEN

    # Compute parameter values X(k) (AE/DO section in SIF)
    # X(-1)=0, X(0)=0, X(1..NFREE)=1, X(NFREE+1..N+2)=0
    xp = zeros(T, N + 5)  # index k -> xp[k + 3] for k in -1..N+2
    for i in 1:NFREE
        xp[i + 3] = T(1)
    end
    xpval(k) = (k < -1 || k > N + 2) ? T(0) : xp[k + 3]

    # Compute L-group linear coefficients for X(I), I=1..N
    # For I=1..NFND: Q = -6*xp(I) + 4*xp(I+1) + 4*xp(I-1) - xp(I+2) - xp(I-2)
    # For I=NFND+1..N: same + 1.0
    l_coefs = zeros(T, N)
    for I in 1:NFND
        l_coefs[I] = -T(6)*xpval(I) + T(4)*xpval(I+1) + T(4)*xpval(I-1) -
                      xpval(I+2) - xpval(I-2)
    end
    for I in NFND+1:N
        l_coefs[I] = -T(6)*xpval(I) + T(4)*xpval(I+1) + T(4)*xpval(I-1) -
                      xpval(I+2) - xpval(I-2) + T(1)
    end

    # Variable indexing: X(I) -> Julia index I, for I=1..N
    lvar = zeros(T, N)          # default lvar=0 (SIF default)
    uvar = fill(T(Inf), N)
    start_vals = fill(T(0.5), N)

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, N; lvar = lvar, uvar = uvar, start = start_vals)

    # Objective: GROUP TYPE HALFL2 = phi(u) = 0.5*u^2 (no scale, so contribution = 0.5*v^2)
    # Q(0): v = x[1]
    ExaModels.@add_obj(ec, T(0.5) * x[1]^2)

    # Q(1): v = 2*x[1] - x[2]
    ExaModels.@add_obj(ec, T(0.5) * (T(2)*x[1] - x[2])^2)

    # Q(I) for I=2..N-1: v = x[I+1] + x[I-1] - 2*x[I]
    q_interior = [(xi = i, xim1 = i - 1, xip1 = i + 1) for i in 2:N-1]
    ExaModels.@add_obj(ec,
        T(0.5) * (x[d.xip1] + x[d.xim1] - T(2)*x[d.xi])^2
        for d in q_interior)

    # Q(N): v = 2*x[N] - x[N-1]
    ExaModels.@add_obj(ec, T(0.5) * (T(2)*x[N] - x[N-1])^2)

    # Q(N+1): v = x[N]
    ExaModels.@add_obj(ec, T(0.5) * x[N]^2)

    # L group (identity type): contribution = sum_I l_coef[I] * x[I]
    # Only add nonzero coefficients
    l_data = [(xi = i, coef = l_coefs[i]) for i in 1:N if l_coefs[i] != T(0)]
    if !isempty(l_data)
        ExaModels.@add_obj(ec, d.coef * x[d.xi] for d in l_data)
    end

    return ExaModels.ExaModel(ec; kwargs...)
end

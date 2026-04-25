# BRATU1D: Bratu's 1D problem (N=5001, unconstrained)
# Source: Buckley (1989), problem 121; SIF input: Ph. Toint, Dec 1989
# classification OXR2-MN-V-0
#
# N interior points X(1)..X(N), plus boundary X(0)=X(N+1)=0 (fixed).
# Total nvar = N+2 = 5003, var k = X(k-1) for k=1..N+2.
# X(0)=var1 and X(N+1)=var(N+2) are fixed at 0.
#
# Parameters: LAMBDA=-3.4, H=1/(N+1), 2/H=(N+1)*2, LH=H*LAMBDA, 2LH=2*LH
#
# Objective = GC(0) + sum_{I=1}^{N} [GA(I) + GB(I) + GC(I)]
#   GC(I) = 2LH * BRA(X(I), X(I+1)) = 2LH * (exp(X(I+1))-exp(X(I)))/(X(I+1)-X(I))
#   GA(I) = (2/H) * X(I)^2
#   GB(I) = (-2/H) * X(I) * X(I-1)
#
# Start: X(0)=0, X(I) = -0.1*I^2/(N+1) for I=1..N, X(N+1)=0.
function bratu1d_model(N = 5001; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(-3.4)
    H = T(1) / T(N + 1)
    twoLH = T(2) * H * LAMBDA           # 2 * H * LAMBDA (negative)
    twooverH = T(2) / H                  # 2/H = 2*(N+1)

    # Variable indexing: var k = X(k-1), k = 1..N+2
    # X(0) = var 1 (fixed), X(I) = var I+1 for I=1..N, X(N+1) = var N+2 (fixed)
    nvar = N + 2
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    lvar[1] = T(0); uvar[1] = T(0)          # X(0) = 0
    lvar[nvar] = T(0); uvar[nvar] = T(0)    # X(N+1) = 0

    # Starting point: X(I) = -0.1 * I^2 / (N+1)
    start = Vector{T}(undef, nvar)
    start[1] = T(0)
    for I in 1:N
        start[I+1] = T(-0.1) * T(I)^2 / T(N + 1)
    end
    start[nvar] = T(0)

    # BRA element data: pairs (X(I), X(I+1)) for I=0..N → var indices (I+1, I+2)
    bra_data = [(a = I + 1, b = I + 2) for I in 0:N]

    # GA+GB element data: (X(I), X(I-1)) for I=1..N → var indices (I+1, I)
    gab_data = [(k = I + 1, km = I) for I in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # GC(I) for I=0..N: twoLH * (exp(X(I+1)) - exp(X(I))) / (X(I+1) - X(I))
    ExaModels.@add_obj(
        c,
        twoLH * (exp(x[d.b]) - exp(x[d.a])) / (x[d.b] - x[d.a])
        for d in bra_data
    )

    # GA(I): (2/H) * X(I)^2   and   GB(I): (-2/H) * X(I) * X(I-1)
    ExaModels.@add_obj(
        c,
        twooverH * x[d.k]^2 - twooverH * x[d.k] * x[d.km]
        for d in gab_data
    )

    return ExaModels.ExaModel(c; kwargs...)
end

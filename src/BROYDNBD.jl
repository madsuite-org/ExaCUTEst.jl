# BROYDNBD: Broyden banded system of nonlinear equations (least squares)
# N=5000, KAPPA1=2, KAPPA2=5, KAPPA3=1, LB=5, UB=1
# All FREE, start=1.0; N equality constraints (NOR classification)
#
# G(I) = KAPPA1*x[I] + KAPPA2*x[I]^3
#        - KAPPA3 * sum_{j in nbrs(I)} (x[j] + x[j]^2)
# where nbrs(I) = {max(1,I-LB)..I-1} ∪ {I+1..min(N,I+UB)}
#
# Use data array with precomputed (i, lower, upper) bounds for neighbor sums.
function broydnbd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    KAPPA1 = T(2)
    KAPPA2 = T(5)
    KAPPA3 = T(1)
    LB = 5
    UB = 1

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))

    # Main bulk: I=LB+1..N-UB-1 (1-based: i=LB+1..N-UB, same range)
    # For these, lower neighbor range is i-LB..i-1 (LB=5 terms), upper is i+1..i+UB (UB=1 term)
    # Unroll the fixed-bandwidth sum: 5 left + 1 right = 6 neighbors
    ExaModels.@add_con(
        c,
        KAPPA1 * x[i] + KAPPA2 * x[i]^3
        - KAPPA3 * (x[i-5] + x[i-4] + x[i-3] + x[i-2] + x[i-1] + x[i+1])
        - KAPPA3 * (x[i-5]^2 + x[i-4]^2 + x[i-3]^2 + x[i-2]^2 + x[i-1]^2 + x[i+1]^2)
        for i in (LB+1):(N-UB)
    )

    # Boundary constraints: I=1..LB (fewer left neighbors)
    for i in 1:LB
        lo = 1          # no left neighbors below 1
        hi = min(N, i + UB)
        nbrs = [j for j in lo:hi if j != i]
        nbrs_data = [(j = j,) for j in nbrs]
        ExaModels.@add_con(
            c,
            KAPPA1 * x[i] + KAPPA2 * x[i]^3
            - KAPPA3 * sum(x[d.j] + x[d.j]^2 for d in nbrs_data)
        )
    end

    # Boundary constraints: I=N-UB+1..N (fewer right neighbors), when UB>0
    for i in (N-UB+1):N
        lo = max(1, i - LB)
        hi = N           # no right neighbors beyond N
        nbrs = [j for j in lo:hi if j != i]
        nbrs_data = [(j = j,) for j in nbrs]
        ExaModels.@add_con(
            c,
            KAPPA1 * x[i] + KAPPA2 * x[i]^3
            - KAPPA3 * sum(x[d.j] + x[d.j]^2 for d in nbrs_data)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end

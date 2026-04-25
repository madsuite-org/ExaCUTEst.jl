# LUKVLE10: Generalized Brown function with Broyden tridiagonal constraints
# Source: Problem 5.10, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/2=5000 pairs, N-2=9998 equality constraints
# All variables free; start x[odd]=-1, x[even]=1
#
# CUTEst variable ordering (empirical, from sifdecoder constraint loop):
#   Same structure as LUKVLE8: constraint loop C(K) introduces X(K+1) first,
#   then X(K), giving CUTEst pos 1=SIF X(N-1), pos N-1=SIF X(1), identity elsewhere.
#
# Objective:
#   sum_{i=1}^{N/2} (x[2i-1]^2)^(x[2i]^2+1)     (OBJ1, identity group)
#   + sum_{i=1}^{N/2} (x[2i]^2)^(x[2i-1]^2+1)   (OBJ2, identity group)
#
# ELEMENT NASTY(X,Y): F = (X^2)^(Y^2+1)
#
# Constraints C(k) for k=1..N-2 (equality = 0):
#   3*x[k+1] - x[k] - 2*x[k+2] - 2*x[k+1]^2 + 1 = 0
function lukvle10_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    # Build SIF→CUTEst permutation: perm[k] = CUTEst position for SIF X(k)
    # Constraint loop C(K) introduces X(K+1) first, so X(2) gets pos 1, X(1) gets pos 2, etc.
    # Net result: swap SIF X(1) ↔ SIF X(N-1) in CUTEst ordering
    perm = collect(1:N)
    if N >= 2
        perm[1], perm[N-1] = N - 1, 1
    end

    # Start: SIF X(odd)=-1, X(even)=1. Convert to CUTEst ordering.
    # start_cutest[c] = value of SIF X(inv_perm[c]) = SIF X(perm[c]) (perm is self-inverse)
    start = T[isodd(perm[c]) ? T(-1) : T(1) for c in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: NASTY(X(2i-1), X(2i)) + NASTY(X(2i), X(2i-1))
    # SIF pair i=1: (X(1), X(2)) → CUTEst (perm[1], perm[2]) = (N-1, 2)
    # SIF pair i=N2: (X(N-1), X(N)) → CUTEst (perm[N-1], perm[N]) = (1, N)
    # SIF pairs i=2..N2-1: (X(2i-1), X(2i)) → identity (no affected indices)
    let a1 = perm[1], b1 = perm[2]  # = N-1, 2
        ExaModels.@add_obj(c, (x[a1]^2)^(x[b1]^2 + T(1)))
        ExaModels.@add_obj(c, (x[b1]^2)^(x[a1]^2 + T(1)))
    end
    # Generic pairs i=2..N2-1
    if N2 >= 3
        ExaModels.@add_obj(c, (x[2i-1]^2)^(x[2i]^2 + T(1)) for i in 2:(N2-1))
        ExaModels.@add_obj(c, (x[2i]^2)^(x[2i-1]^2 + T(1)) for i in 2:(N2-1))
    end
    # Special pair i=N2
    let aN = perm[N-1], bN = perm[N]  # = 1, N
        ExaModels.@add_obj(c, (x[aN]^2)^(x[bN]^2 + T(1)))
        ExaModels.@add_obj(c, (x[bN]^2)^(x[aN]^2 + T(1)))
    end

    # Constraints C(k): 3*X(k+1) - X(k) - 2*X(k+2) - 2*X(k+1)^2 + 1 = 0
    # Must add in order k=1..N-2 to match CUTEst constraint ordering.
    # SIF X(k) → CUTEst perm[k]. Affected: perm[1]=N-1, perm[N-1]=1.
    # C(k) uses X(k), X(k+1), X(k+2). Affected when k=1, k+1=1 (impossible), k+2=1 (impossible),
    # or k=N-1 (out of range), k+1=N-1 (k=N-2), k+2=N-1 (k=N-3).
    # → C(1): X(1)→perm[1]=N-1 (not pos 1)
    # → C(N-3): X(k+2)=X(N-1)→perm[N-1]=1 (not pos N-1)
    # → C(N-2): X(k+1)=X(N-1)→perm[N-1]=1 (not pos N-1)

    let p1 = perm[1]  # = N-1
        # C(1): 3*X(2) - X(1) - 2*X(3) - 2*X(2)^2 + 1
        ExaModels.@add_con(
            c,
            T(3) * x[2] - x[p1] - T(2) * x[3] - T(2) * x[2]^2 + T(1),
        )
    end

    # Generic constraints k=2..N-4 (identity mapping)
    if N >= 7
        ExaModels.@add_con(
            c,
            T(3) * x[k+1] - x[k] - T(2) * x[k+2] - T(2) * x[k+1]^2 + T(1) for
            k in 2:(N-4)
        )
    end

    let pNm1 = perm[N-1]  # = 1
        # C(N-3): 3*X(N-2) - X(N-3) - 2*X(N-1) - 2*X(N-2)^2 + 1
        ExaModels.@add_con(
            c,
            T(3) * x[N-2] - x[N-3] - T(2) * x[pNm1] - T(2) * x[N-2]^2 + T(1),
        )
        # C(N-2): 3*X(N-1) - X(N-2) - 2*X(N) - 2*X(N-1)^2 + 1
        ExaModels.@add_con(
            c,
            T(3) * x[pNm1] - x[N-2] - T(2) * x[N] - T(2) * x[pNm1]^2 + T(1),
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end

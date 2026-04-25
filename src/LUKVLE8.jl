# LUKVLE8: Augmented Lagrangian function with discrete boundary value constraints
# Source: Problem 5.8, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/5=2000 objective groups, N-2=9998 equality constraints
# All variables free; start x[odd]=-1, x[even]=2
#
# Parameters:
#   LAMBDA1=-0.002008, LAMBDA2=-0.001900, LAMBDA3=-0.000261
#   H=1/(N+1), H^2/2 = 0.5/((N+1)^2)
#   LAMBA1+10 = LAMBDA1+10 = 9.997992
#   LAMBA3-1 = LAMBDA3-1 = -1.000261
#
# CUTEst variable ordering (empirical, from sifdecoder constraint loop):
#   CUTEst pos 1   = SIF X(N-1)
#   CUTEst pos N-1 = SIF X(1)
#   CUTEst pos k   = SIF X(k)   for all k ≠ 1 and k ≠ N-1
#
# Objective terms (block i=1..N/5, 5 vars per block: X(5i-4..5i)):
#   O1(i): exp(X(5i)*X(5i-1)*X(5i-2)*X(5i-3)*X(5i-4))
#   O2(i): 10*(X(5i)^2+X(5i-1)^2+...+X(5i-4)^2 - (LAMBDA1+10))^2
#   O3(i): 10*(X(5i-3)*X(5i-2) - 5*X(5i-1)*X(5i) - LAMBDA2)^2
#   O4(i): 10*(X(5i-4)^3+X(5i-3)^3 - (LAMBDA3-1))^2
#
# Constraints C(k) for k=1..N-2 (equality = 0):
#   2*X(k+1) - X(k) - X(k+2) + H^2/2*(X(k+1) + h*(k+1)+1)^2 = 0
function lukvle8_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N % 5 == 0 "N must be divisible by 5"
    N5 = N ÷ 5
    N2 = N - 2

    lambda1 = T(-0.002008)
    lambda2 = T(-0.001900)
    lambda3 = T(-0.000261)
    lamba1p10 = lambda1 + T(10)   # LAMBA1+10 = 9.997992
    lamba3m1 = lambda3 - T(1)     # LAMBA3-1 = -1.000261

    h = T(1) / T(N + 1)
    h2h2 = T(0.5) * h^2           # H^2/2

    # Build SIF→CUTEst permutation: perm[k] = CUTEst position for SIF X(k)
    # Swap SIF X(1) ↔ SIF X(N-1) in CUTEst ordering
    perm = collect(1:N)
    if N >= 2
        perm[1], perm[N-1] = N - 1, 1
    end
    # For use as CUTEst positions in expressions:
    # p[k] = CUTEst position of SIF X(k)
    p = perm

    # Start: odd SIF indices -> -1, even SIF indices -> 2
    # start_cutest[c] = value of variable at CUTEst position c
    # CUTEst pos c = SIF X(k) where k = inv_perm[c] = perm[c] (since perm is its own inverse here)
    # Actually perm is a self-inverse (swap of two elements), so inv_perm = perm
    start = T[isodd(perm[c]) ? T(-1) : T(2) for c in 1:N]

    # Precompute P[k] = h*(k+1)+1 for constraint C(k) SHIFTSQR element
    P = T[(k + 1) * h + T(1) for k in 1:N2]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, P, P)

    # Objective: O1 terms (group type EXP: exp(v))
    # For generic blocks i=2..N5-1 (avoid first and last which have permuted vars)
    # First block i=1: X(1..5), where X(1) is at CUTEst pos N-1=p[1], X(2..5) at pos 2..5
    # Last block i=N5: X(5N5-4..5N5) = X(N-4..N), where X(N-1) is at CUTEst pos 1=p[N-1]

    # Generic blocks i=2..N5-1 (X(5i-4..5i), all mapped by identity since neither 1 nor N-1)
    if N5 >= 3
        ExaModels.@add_obj(
            c,
            exp(x[5i] * x[5i-1] * x[5i-2] * x[5i-3] * x[5i-4]) for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) *
            (x[5i]^2 + x[5i-1]^2 + x[5i-2]^2 + x[5i-3]^2 + x[5i-4]^2 - lamba1p10)^2
            for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[5i-3] * x[5i-2] - T(5) * x[5i-1] * x[5i] - lambda2)^2
            for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[5i-4]^3 + x[5i-3]^3 - lamba3m1)^2 for i in 2:(N5-1)
        )
    end

    # First block i=1: X(1)=x[p[1]], X(2)=x[2], X(3)=x[3], X(4)=x[4], X(5)=x[5]
    let p1 = p[1]
        ExaModels.@add_obj(c, exp(x[5] * x[4] * x[3] * x[2] * x[p1]))
        ExaModels.@add_obj(
            c,
            T(10) * (x[5]^2 + x[4]^2 + x[3]^2 + x[2]^2 + x[p1]^2 - lamba1p10)^2,
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[2] * x[3] - T(5) * x[4] * x[5] - lambda2)^2,
        )
        ExaModels.@add_obj(c, T(10) * (x[p1]^3 + x[2]^3 - lamba3m1)^2)
    end

    # Last block i=N5: X(N-4..N) = X(5N5-4..5N5)
    # In block i=N5: 5i=N, 5i-1=N-1, 5i-2=N-2, 5i-3=N-3, 5i-4=N-4
    # X(N-1) is at CUTEst pos p[N-1]=1; all others are identity
    let pNm1 = p[N-1]
        ExaModels.@add_obj(c, exp(x[N] * x[pNm1] * x[N-2] * x[N-3] * x[N-4]))
        ExaModels.@add_obj(
            c,
            T(10) *
            (x[N]^2 + x[pNm1]^2 + x[N-2]^2 + x[N-3]^2 + x[N-4]^2 - lamba1p10)^2,
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[N-3] * x[N-2] - T(5) * x[pNm1] * x[N] - lambda2)^2,
        )
        ExaModels.@add_obj(c, T(10) * (x[N-4]^3 + x[N-3]^3 - lamba3m1)^2)
    end

    # Constraints C(k): 2*X(k+1) - X(k) - X(k+2) + H^2/2*(X(k+1)+P[k])^2 = 0
    # Constraints must be added in order k=1..N-2 to match CUTEst ordering.
    # Special cases involve SIF X(1) (CUTEst pos N-1=p[1]) and SIF X(N-1) (CUTEst pos 1=p[N-1]):
    #   C(1): X(1) at pos p[1]=N-1
    #   C(N-3): X(K+2)=X(N-1) at pos p[N-1]=1
    #   C(N-2): X(K+1)=X(N-1) at pos p[N-1]=1

    # C(1): k=1, X(1)→p[1]=N-1, X(2)→2, X(3)→3
    let p1 = p[1]
        ExaModels.@add_con(
            c,
            T(2) * x[2] - x[p1] - x[3] + h2h2 * (x[2] + P[1])^2,
        )
    end

    # Generic constraints: k=2..N-4 (identity mapping, no boundary variables involved)
    if N >= 7
        ExaModels.@add_con(
            c,
            T(2) * x[k+1] - x[k] - x[k+2] + h2h2 * (x[k+1] + P[k])^2 for
            k in 2:(N-4)
        )
    end

    # C(N-3): k=N-3, X(N-3)→N-3, X(N-2)→N-2, X(N-1)→p[N-1]=1
    let pNm1 = p[N-1]
        ExaModels.@add_con(
            c,
            T(2) * x[N-2] - x[N-3] - x[pNm1] + h2h2 * (x[N-2] + P[N-3])^2,
        )
    end

    # C(N-2): k=N-2, X(N-2)→N-2, X(N-1)→p[N-1]=1, X(N)→N
    let pNm1 = p[N-1]
        ExaModels.@add_con(
            c,
            T(2) * x[pNm1] - x[N-2] - x[N] + h2h2 * (x[pNm1] + P[N-2])^2,
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end

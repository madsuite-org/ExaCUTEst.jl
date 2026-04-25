# LUKVLE7: Trigonometric tridiagonal function with simplified five-diagonal constraints
# Source: Problem 5.7, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 4 equality constraints
# All variables free; start x[i]=1
#
# CUTEst variable ordering (empirical, due to sifdecoder reordering):
#   CUTEst pos 1..4   = SIF X(1..4)
#   CUTEst pos 5      = SIF X(N)
#   CUTEst pos 6      = SIF X(N-1)
#   CUTEst pos 7      = SIF X(N-2)
#   CUTEst pos 8..N-3 = SIF X(8..N-3)
#   CUTEst pos N-2    = SIF X(7)
#   CUTEst pos N-1    = SIF X(6)
#   CUTEst pos N      = SIF X(5)
#
# Objective (single identity group):
#   sum_{k=1}^{N} (-k)*cos(X(k)) + sum_{k=1}^{N-1} 2*sin(X(k)) - (N-1)*sin(X(N)) + N*(N+1)/2
#
# ELEMENT CUBEP: V^3 - V*W
# ELEMENT SQR: V^2
#
# Constraints (using CUTEst positions):
#   C(1): 4*x[1]+x[2] - 4*x[2]^2 - x[3]^2 = 0
#   C(2): 6*x[2]+x[3] + 8*(x[2]^3-x[2]*x[1]) - 4*x[3]^2 - x[4]^2 - 2 = 0
#   C(3): uses CUTEst x[6]=X(N-1), x[7]=X(N-2), x[5]=X(N)
#         6*x[6]+x[N-3_cutest] + 8*(x[6]^3-x[6]*x[7]) - 4*x[5]^2 + x[7]^2 - 2 = 0
#         where X(N-3) is at CUTEst position N-3 (for N-3 ≥ 8, i.e. N ≥ 11)
#   C(4): uses CUTEst x[5]=X(N), x[6]=X(N-1), x[7]=X(N-2)
#         2*x[5] - x[7] + 8*(x[5]^3-x[5]*x[6]) + x[6]^2 = 0
#
# For the constraints, we need CUTEst positions of X(N-3) and X(N-2):
#   X(N-3): for N-3 >= 8 (N >= 11), CUTEst pos = N-3; for N-3 in {5,6,7}, special case
#   X(N-2): CUTEst pos 7
function lukvle7_model(N = 10000; T = Float64, backend = nothing, kwargs...)

    # Build SIF→CUTEst permutation: perm[k] = CUTEst position for SIF X(k)
    perm = collect(1:N)
    # Swap {5,6,7} ↔ {N,N-1,N-2}
    if N >= 7
        perm[5], perm[N]   = N,   5
        perm[6], perm[N-1] = N-1, 6
        perm[7], perm[N-2] = N-2, 7
    end
    # inv_perm[c] = SIF index k for CUTEst pos c
    inv_perm = invperm(perm)  # inv_perm[c] = k such that perm[k]=c

    # Start: all 1.0 in SIF order → permuted to CUTEst order
    start = ones(T, N)  # CUTEst pos → value 1.0 (same since all 1.0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Parameters: SIF index k for each CUTEst position c
    # k[c] = inv_perm[c] = SIF index of CUTEst variable c
    k_par = T[T(inv_perm[c]) for c in 1:N]
    ExaModels.@add_par(c, kpar, k_par)

    # SI coefficient for SIF X(k): 2 for k=1..N-1, -(N-1) for k=N
    # In terms of CUTEst pos c: si_coeff[c] = 2 if inv_perm[c] < N, else -(N-1)
    si_coeff = T[inv_perm[c] < N ? T(2) : T(-(N - 1)) for c in 1:N]
    ExaModels.@add_par(c, sipar, si_coeff)

    # Objective terms:
    # CO terms: -k*cos(x[c]) for all c (k = kpar[c])
    # SI terms: sipar[c]*sin(x[c]) for all c
    # Constant: N*(N+1)/2, distributed as (N+1)/2 per CO term
    half_Np1 = T(N + 1) / T(2)
    ExaModels.@add_obj(c, -kpar[i] * cos(x[i]) + sipar[i] * sin(x[i]) + half_Np1 for i in 1:N)

    # Constraints use CUTEst positions directly:
    # CUTEst pos of SIF variables:
    #   X(1)=pos1, X(2)=pos2, X(3)=pos3, X(4)=pos4
    #   X(N-3)=pos(N-3) for N-3 >= 8 (assumed N >= 11)
    #   X(N-2)=pos7, X(N-1)=pos6, X(N)=pos5
    p_xNm2 = perm[N-2]  # = 7
    p_xNm1 = perm[N-1]  # = 6
    p_xN   = perm[N]    # = 5
    p_xNm3 = perm[N-3]  # = N-3 for N >= 11

    # C(1): 4*X(1)+X(2) - 4*X(2)^2 - X(3)^2 = 0
    ExaModels.@add_con(c, T(4) * x[1] + x[2] - T(4) * x[2]^2 - x[3]^2)

    # C(2): 6*X(2)+X(3) + 8*(X(2)^3-X(2)*X(1)) - 4*X(3)^2 - X(4)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[2] + x[3] + T(8) * (x[2]^3 - x[2] * x[1]) - T(4) * x[3]^2 - x[4]^2 -
        T(2),
    )

    # C(3): 6*X(N-1)+(-X(N-3)) + 8*(X(N-1)^3-X(N-1)*X(N-2)) - 4*X(N)^2 + X(N-2)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[p_xNm1] - x[p_xNm3] + T(8) * (x[p_xNm1]^3 - x[p_xNm1] * x[p_xNm2]) -
        T(4) * x[p_xN]^2 + x[p_xNm2]^2 - T(2),
    )

    # C(4): 2*X(N)+(-X(N-2)) + 8*(X(N)^3-X(N)*X(N-1)) + X(N-1)^2 = 0
    ExaModels.@add_con(
        c,
        T(2) * x[p_xN] - x[p_xNm2] + T(8) * (x[p_xN]^3 - x[p_xN] * x[p_xNm1]) +
        x[p_xNm1]^2,
    )

    return ExaModels.ExaModel(c; kwargs...)
end

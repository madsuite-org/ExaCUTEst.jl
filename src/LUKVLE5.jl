# LUKVLE5: Generalized Broyden tridiagonal function with five-diagonal constraints
# Source: Problem 5.5, Luksan & Vlcek (1999), TR 767
# N+2=10002 variables (SIF X(0..N+1)), N-4=9996 equality constraints
# Bounds: X(0)=X(N+1)=0 (fixed); all others free; start: X(1..N)=-1
#
# CUTEst variable ordering (due to sifdecoder reordering):
#   SIF X(N)   → CUTEst pos 1
#   SIF X(N-1) → CUTEst pos 2
#   SIF X(k)   → CUTEst pos k+1  for k=2..N-2
#   SIF X(1)   → CUTEst pos N
#   SIF X(0)   → CUTEst pos N+1  (fixed=0)
#   SIF X(N+1) → CUTEst pos N+2  (fixed=0)
#
# Objective (L7/3 group type: |v|^(7/3)):
#   sum_{I=1}^{N} |3*X(I) - X(I+1) - X(I-1) - 2*X(I)^2 + 1|^(7/3)
#   (with X(0)=X(N+1)=0)
#
# Constraints C(K) for K=1..N-4:
#   6*X(K+2) - X(K) + X(K+3) + 8*(X(K+2)^3 - X(K+2)*X(K+1)) - 4*X(K+3)^2
#   + X(K+1)^2 - X(K+4)^2 - 2 = 0
function lukvle5_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N >= 5 "N must be >= 5"
    Nvar = N + 2  # total CUTEst variables including fixed X(0) and X(N+1)

    # Permutation: perm[k] = CUTEst position of SIF X(k), k=0..N+1
    # perm[N] = 1, perm[N-1] = 2, perm[k] = k+1 for k=2..N-2, perm[1] = N
    # perm[0] = N+1 (fixed), perm[N+1] = N+2 (fixed)
    function perm(k)
        if k == 0; return N + 1
        elseif k == 1; return N
        elseif k == N - 1; return 2
        elseif k == N; return 1
        elseif k == N + 1; return N + 2
        else; return k + 1  # k=2..N-2
        end
    end

    # Start: X(1..N)=-1, X(0)=X(N+1)=0 (fixed via bounds)
    start = zeros(T, Nvar)
    for k in 1:N
        start[perm(k)] = T(-1)
    end
    # Fixed positions already 0 (zeros initialization)

    # Bounds: X(0) and X(N+1) fixed at 0
    lvar = fill(T(-Inf), Nvar)
    uvar = fill(T(Inf), Nvar)
    lvar[perm(0)] = T(0)
    uvar[perm(0)] = T(0)
    lvar[perm(N+1)] = T(0)
    uvar[perm(N+1)] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, Nvar; lvar = lvar, uvar = uvar, start = start)

    # Build parameter arrays for CUTEst positions needed by objective and constraints
    # Objective OBJ(I) for I=1..N: uses X(I-1), X(I), X(I+1)
    # CUTEst pos arrays:
    pI  = T[T(perm(I))   for I in 1:N]   # pos of X(I)
    pIm1 = T[T(perm(I-1)) for I in 1:N]  # pos of X(I-1)
    pIp1 = T[T(perm(I+1)) for I in 1:N]  # pos of X(I+1)
    ExaModels.@add_par(c, posI,   pI)
    ExaModels.@add_par(c, posIm1, pIm1)
    ExaModels.@add_par(c, posIp1, pIp1)

    # Objective: |v|^(7/3) where v = 3*X(I) - X(I+1) - X(I-1) - 2*X(I)^2 + 1
    ExaModels.@add_obj(
        c,
        abs(T(3) * x[posI[i]] - x[posIp1[i]] - x[posIm1[i]] - T(2) * x[posI[i]]^2 + T(1))^T(7/3)
        for i in 1:N
    )

    # Constraints C(K) for K=1..N-4: uses X(K),X(K+1),X(K+2),X(K+3),X(K+4)
    # For K=1..N-4: K+4 <= N, so all indices in range
    # Boundary case: K=1 uses X(1)→perm[1]=N; for K up to N-4 uses X(K+4)=X(N)→1
    pK  = T[T(perm(K))   for K in 1:N-4]
    pK1 = T[T(perm(K+1)) for K in 1:N-4]
    pK2 = T[T(perm(K+2)) for K in 1:N-4]
    pK3 = T[T(perm(K+3)) for K in 1:N-4]
    pK4 = T[T(perm(K+4)) for K in 1:N-4]
    ExaModels.@add_par(c, posK,  pK)
    ExaModels.@add_par(c, posK1, pK1)
    ExaModels.@add_par(c, posK2, pK2)
    ExaModels.@add_par(c, posK3, pK3)
    ExaModels.@add_par(c, posK4, pK4)

    ExaModels.@add_con(
        c,
        T(6) * x[posK2[k]] - x[posK[k]] + x[posK3[k]] +
        T(8) * (x[posK2[k]]^3 - x[posK2[k]] * x[posK1[k]]) -
        T(4) * x[posK3[k]]^2 + x[posK1[k]]^2 - x[posK4[k]]^2 - T(2)
        for k in 1:(N-4)
    )

    return ExaModels.ExaModel(c; kwargs...)
end

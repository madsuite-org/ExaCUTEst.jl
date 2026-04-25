# LUKVLE4: Chained Cragg and Levy problem with tridiagonal constraints
# Source: Problem 5.4, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/2-1=4999 equality constraints
# All variables free; start: X(4k+1)=1, X(4k+2..4k+4)=2
#
# CUTEst variable ordering (due to sifdecoder reordering):
#   j(m) = 10001-m  for m ≠ 5000, 5001  (reverse permutation)
#   j(5000) = 5000  (identity)
#   j(5001) = 5001  (identity)
#
# Objective (groups A,B,C,D,F for I=1..N/2-1=4999, all L4 or AL6 or L8 or L2 types):
#   A(I): L4, v = -X(2I) + exp(X(2I-1));  contrib = v^4
#   B(I): AL6, v = X(2I) - X(2I+1);        contrib = 100*v^6
#   C(I): L4, v = (X(2I+1)-X(2I+2)) + tan(X(2I+1)-X(2I+2))
#              + 6*X(I+1) + 8*(X(I+1)^3 - X(I+1)*X(I)) - 4*X(I+2)^2 - 2
#         contrib = v^4
#   D(I): L8, v = X(2I-1);  contrib = v^8
#   F(I): L2, v = X(2I+2) - 1;  contrib = v^2
#
# Note: C(I) group involves BOTH the "Wood function" pairs X(2I+1),X(2I+2) (from outer loop)
# AND the "tridiagonal" triplet X(I),X(I+1),X(I+2) (from the K=I inner loop).
#
# Constraints C(K) for K=5000..9998 (ci = K-4999 = 1..4999):
#   6*X(K+1) + 8*(X(K+1)^3 - X(K+1)*X(K)) - 4*X(K+2)^2 - 2 = 0
#   using CUTEst positions: j(K), j(K+1), j(K+2)
function lukvle4_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    @assert N >= 6 "N must be >= 6"
    N2m1 = N ÷ 2 - 1  # = N/2 - 1

    # CUTEst permutation: j(m) = N+1-m for m ≠ N/2, N/2+1; j(N/2) = N/2; j(N/2+1) = N/2+1
    # For N=10000: j(m)=10001-m except j(5000)=5000, j(5001)=5001
    half = N ÷ 2
    function j(m)
        if m == half; return half
        elseif m == half + 1; return half + 1
        else; return N + 1 - m
        end
    end

    # Start: in SIF order X(4k+1)=1, X(4k+2)=X(4k+3)=X(4k+4)=2
    # CUTEst pos c = j(m) for SIF X(m): start[c] = sif_start[m]
    perm = [j(m) for m in 1:N]
    sif_start = T[((m-1) % 4 == 0) ? T(1) : T(2) for m in 1:N]
    start = zeros(T, N)
    for m in 1:N
        start[perm[m]] = sif_start[m]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: groups for I=1..N/2-1
    # A(I): v = -X(2I) + exp(X(2I-1)) = -x[j(2I)] + exp(x[j(2I-1)])
    # B(I): v = X(2I) - X(2I+1) = x[j(2I)] - x[j(2I+1)]
    # C(I): v = (X(2I+1)-X(2I+2)) + tan(X(2I+1)-X(2I+2))
    #          + 6*X(I+1) + 8*(X(I+1)^3 - X(I+1)*X(I)) - 4*X(I+2)^2 - 2
    # D(I): v = X(2I-1) = x[j(2I-1)]
    # F(I): v = X(2I+2) - 1 = x[j(2I+2)] - 1

    # Precompute CUTEst positions for each group's SIF indices
    # For I=1..N/2-1: indices 2I-1, 2I, 2I+1, 2I+2, I, I+1, I+2
    # These all need to go through j(). Since j(m)=N+1-m is a formula, we can
    # either precompute or use @add_par. We'll handle it explicitly.

    # All groups use j(m) = N+1-m in general. The only exceptions are j(N/2) and j(N/2+1).
    # For I=1..N/2-1:
    #   2I ranges 2..N-2, 2I-1 ranges 1..N-3, 2I+1 ranges 3..N-1, 2I+2 ranges 4..N
    #   None of these equal N/2 or N/2+1 for most I.
    #   Exception: when 2I-1=N/2 → I=(N/2+1)/2 (non-integer for even N/2) → no issue
    #              2I=N/2 → I=N/4 → j(N/2)=N/2 (identity)
    #              2I+1=N/2+1 → I=N/4 → j(N/2+1)=N/2+1 (identity)
    #   So exceptions only affect groups where 2I=N/2 or 2I-1=N/2 or 2I+1=N/2 etc.
    #   For I, I+1, I+2: when I=N/2 or I=N/2+1 (impossible since I<=N/2-1)
    #   All relevant SIF indices i and i+k for the "tridiagonal" part (I, I+1, I+2):
    #   max value = N/2+1 = 5001. j(5001)=5001 (identity). j(5000)=5000. Fine.

    # Strategy: use @add_par with precomputed CUTEst positions
    # Build arrays of positions for each term:
    jA1 = T[T(j(2I-1)) for I in 1:N2m1]  # j(2I-1): for D and A
    jA2 = T[T(j(2I))   for I in 1:N2m1]  # j(2I): for A, B
    jA3 = T[T(j(2I+1)) for I in 1:N2m1]  # j(2I+1): for B, C tang
    jA4 = T[T(j(2I+2)) for I in 1:N2m1]  # j(2I+2): for C tang, F
    jI0 = T[T(j(I))    for I in 1:N2m1]  # j(I): for C cubic
    jI1 = T[T(j(I+1))  for I in 1:N2m1]  # j(I+1): for C cubic
    jI2 = T[T(j(I+2))  for I in 1:N2m1]  # j(I+2): for C cubic

    ExaModels.@add_par(c, pA1, jA1)   # CUTEst pos of SIF X(2I-1)
    ExaModels.@add_par(c, pA2, jA2)   # CUTEst pos of SIF X(2I)
    ExaModels.@add_par(c, pA3, jA3)   # CUTEst pos of SIF X(2I+1)
    ExaModels.@add_par(c, pA4, jA4)   # CUTEst pos of SIF X(2I+2)
    ExaModels.@add_par(c, pI0, jI0)   # CUTEst pos of SIF X(I)
    ExaModels.@add_par(c, pI1, jI1)   # CUTEst pos of SIF X(I+1)
    ExaModels.@add_par(c, pI2, jI2)   # CUTEst pos of SIF X(I+2)

    # A(I): contrib = v_A^4 where v_A = -x[j(2I)] + exp(x[j(2I-1)])
    ExaModels.@add_obj(c, (-x[pA2[i]] + exp(x[pA1[i]]))^4 for i in 1:N2m1)

    # B(I): contrib = 100 * v_B^6 where v_B = x[j(2I)] - x[j(2I+1)]
    ExaModels.@add_obj(c, T(100) * (x[pA2[i]] - x[pA3[i]])^6 for i in 1:N2m1)

    # C(I): contrib = v_C^4
    ExaModels.@add_obj(
        c,
        let u = x[pA3[i]] - x[pA4[i]]
            (u + tan(u) + T(6) * x[pI1[i]] + T(8) * (x[pI1[i]]^3 - x[pI1[i]] * x[pI0[i]]) -
            T(4) * x[pI2[i]]^2 - T(2))^4
        end for i in 1:N2m1
    )

    # D(I): contrib = v_D^8 where v_D = x[j(2I-1)]
    ExaModels.@add_obj(c, x[pA1[i]]^8 for i in 1:N2m1)

    # F(I): contrib = v_F^2 where v_F = x[j(2I+2)] - 1
    ExaModels.@add_obj(c, (x[pA4[i]] - T(1))^2 for i in 1:N2m1)

    # Constraints C(K=5000..N-2) = ci=1..N/2-1:
    # K = ci + N/2 - 1 = ci + 4999
    # CUTEst pos: j(K), j(K+1), j(K+2) = j(ci+4999), j(ci+5000), j(ci+5001)
    # j(ci+4999) = N+1-(ci+4999) = 5001-ci (for ci>=1, 5001-ci <= 5000, none is 5000 or 5001 for ci>=1)
    # j(ci+5000) = N+1-(ci+5000) = 5000-ci (for ci>=1... but j(5000)=5000, and ci+5000=5000 when ci=0)
    # For ci=1: j(5000)=5000, j(5001)=5001, j(5002)=4999
    # For ci=2..N/2-1: j(ci+4999)=5001-ci, j(ci+5000)=5000-ci, j(ci+5001)=4999-ci
    # All these are >= 1 for ci <= 4999 and != 5000 or 5001 for ci >= 2.

    # Build constraint position arrays
    jC0 = T[T(j(i + half - 1))    for i in 1:N2m1]
    jC1 = T[T(j(i + half))        for i in 1:N2m1]
    jC2 = T[T(j(i + half + 1))    for i in 1:N2m1]
    ExaModels.@add_par(c, pC0, jC0)
    ExaModels.@add_par(c, pC1, jC1)
    ExaModels.@add_par(c, pC2, jC2)

    ExaModels.@add_con(
        c,
        T(6) * x[pC1[k]] + T(8) * (x[pC1[k]]^3 - x[pC1[k]] * x[pC0[k]]) -
        T(4) * x[pC2[k]]^2 - T(2)
        for k in 1:N2m1
    )

    return ExaModels.ExaModel(c; kwargs...)
end

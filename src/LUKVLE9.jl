# LUKVLE9: Modified Brown function with simplified seven-diagonal constraints
# Source: Problem 5.9, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 6 equality constraints
# All variables free; start x[i]=-1
#
# CUTEst variable ordering (empirical, due to sifdecoder reordering):
#   Constraint groups C(4..6) introduce SIF X(N..N-4) at CUTEst positions 7..11,
#   before the OBJ DO loop reaches I=4..6 which would introduce SIF X(7..11).
#   Result:
#     CUTEst pos 7 = SIF X(N), pos 8 = SIF X(N-1), ..., pos 11 = SIF X(N-4)
#     CUTEst pos N-4 = SIF X(7), pos N-3 = SIF X(8), ..., pos N = SIF X(11)
#     All other CUTEst positions: identity (pos k = SIF X(k))
#
# Objective:
#   sum_{i=1}^{N/2} 0.001*(SIF X(2i-1))^2                  (OBJ1, AL2 group type)
#   + sum_{i=1}^{N/2} (-SIF X(2i-1) + SIF X(2i))           (OBJ2, identity group)
#   + sum_{i=1}^{N/2} exp(20*(SIF X(2i-1) - SIF X(2i)))    (OBJ3, EXP20 group type)
#
# Constraints (equality = 0):
# C(1): 4*X(1)+X(2)+X(3) - 4*X(2)^2 - X(3)^2 - X(4)^2 = 0
# C(2): 6*X(2)+X(3)+X(4) + 8*(X(2)^3-X(2)*X(1)) - 4*X(3)^2 + X(1)^2 - X(4)^2 - X(5)^2 - 2 = 0
# C(3): 6*X(3)+X(4)+X(5)-X(1) + 8*(X(3)^3-X(3)*X(2)) - 4*X(4)^2 + X(2)^2 - X(5)^2 + X(1)^2 - X(6)^2 - 2 = 0
# C(4..6): use SIF X(N-5..N) which are at CUTEst pos {9995, 11, 10, 9, 8, 7}
function lukvle9_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    # Build SIF→CUTEst permutation: perm[k] = CUTEst position for SIF X(k)
    # From empirical analysis (constraint Jacobian + Hessian cross terms):
    #   C(4..6) introduce X(N)→7, X(N-1)→8, X(N-2)→9, X(N-3)→10, X(N-4)→11
    #   before OBJ loop I=4..6 which would introduce X(7..12).
    # Full permutation:
    #   perm[7]=N-4, perm[8]=12,  perm[9]=N-2, perm[10]=N-3, perm[11]=N, perm[12]=N-1
    #   perm[N-4]=11, perm[N-3]=10, perm[N-2]=9, perm[N-1]=8, perm[N]=7
    perm = collect(1:N)
    if N >= 12
        perm[7] = N - 4
        perm[8] = 12
        perm[9] = N - 2
        perm[10] = N - 3
        perm[11] = N
        perm[12] = N - 1
        perm[N-4] = 11
        perm[N-3] = 10
        perm[N-2] = 9
        perm[N-1] = 8
        perm[N] = 7
    end

    # Start: all -1 (same in any ordering)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1))

    # Objective: split into identity pairs and permuted pairs.
    # Identity pairs: i=1..3 (SIF X(1..6), all at natural CUTEst pos) and i=7..N2-3 (SIF X(13..N-6))
    # Permuted pairs:
    #   i=4: SIF (7,8) → CUTEst (N-4, N-3) = (perm[7], perm[8])
    #   i=5: SIF (9,10) → CUTEst (N-2, N-1) = (perm[9], perm[10])
    #   i=6: SIF (11,12) → CUTEst (N, 12) = (perm[11], 12)
    #   i=N2-2: SIF (N-5,N-4) → CUTEst (N-5, 11) = (N-5, perm[N-4])
    #   i=N2-1: SIF (N-3,N-2) → CUTEst (10, 9) = (perm[N-3], perm[N-2])
    #   i=N2:   SIF (N-1,N)   → CUTEst (8, 7) = (perm[N-1], perm[N])

    # OBJ1: AL2 group, v = SIF X(2i-1) → 0.001*v^2
    # OBJ2: identity group, v = -SIF X(2i-1) + SIF X(2i)
    # OBJ3: EXP20 group, v = SIF X(2i-1) - SIF X(2i) → exp(20*v)

    # Identity pairs i=1..3
    ExaModels.@add_obj(c, T(0.001) * x[2i-1]^2 for i in 1:3)
    ExaModels.@add_obj(c, -x[2i-1] + x[2i] for i in 1:3)
    ExaModels.@add_obj(c, exp(T(20) * (x[2i-1] - x[2i])) for i in 1:3)

    # Permuted pair i=4: SIF (7,8) → CUTEst (N-4, N-3)
    let a = perm[7], b = perm[8]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Permuted pair i=5: SIF (9,10) → CUTEst (N-2, N-1)
    let a = perm[9], b = perm[10]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Permuted pair i=6: SIF (11,12) → CUTEst (N, N-1) = (perm[11], perm[12])
    let a = perm[11], b = perm[12]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Identity pairs i=7..N2-3 (SIF X(13..N-6), all at natural CUTEst positions)
    if N2 >= 10
        ExaModels.@add_obj(c, T(0.001) * x[2i-1]^2 for i in 7:(N2-3))
        ExaModels.@add_obj(c, -x[2i-1] + x[2i] for i in 7:(N2-3))
        ExaModels.@add_obj(c, exp(T(20) * (x[2i-1] - x[2i])) for i in 7:(N2-3))
    end

    # Permuted pair i=N2-2: SIF (N-5, N-4) → CUTEst (N-5, 11)
    let a = N - 5, b = perm[N-4]  # = 11
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Permuted pair i=N2-1: SIF (N-3, N-2) → CUTEst (10, 9)
    let a = perm[N-3], b = perm[N-2]  # = 10, 9
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Permuted pair i=N2: SIF (N-1, N) → CUTEst (8, 7)
    let a = perm[N-1], b = perm[N]  # = 8, 7
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    # Constraints: use perm[] to map SIF variable indices to CUTEst positions
    # C(1..3) use SIF X(1..6), all identity
    # C(4..6) use SIF X(N-5..N), mapped via perm
    p1 = perm[1]; p2 = perm[2]; p3 = perm[3]; p4 = perm[4]; p5 = perm[5]; p6 = perm[6]
    pN = perm[N]; pNm1 = perm[N-1]; pNm2 = perm[N-2]; pNm3 = perm[N-3]
    pNm4 = perm[N-4]; pNm5 = perm[N-5]

    # C(1): 4*X(1)+X(2)+X(3) - 4*X(2)^2 - X(3)^2 - X(4)^2 = 0
    ExaModels.@add_con(
        c,
        T(4) * x[p1] + x[p2] + x[p3] - T(4) * x[p2]^2 - x[p3]^2 - x[p4]^2,
    )

    # C(2): 6*X(2)+X(3)+X(4) + 8*(X(2)^3-X(2)*X(1)) - 4*X(3)^2 + X(1)^2 - X(4)^2 - X(5)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[p2] + x[p3] + x[p4] + T(8) * (x[p2]^3 - x[p2] * x[p1]) -
        T(4) * x[p3]^2 + x[p1]^2 - x[p4]^2 - x[p5]^2 - T(2),
    )

    # C(3): 6*X(3)+X(4)+X(5)-X(1) + 8*(X(3)^3-X(3)*X(2)) - 4*X(4)^2 + X(2)^2 - X(5)^2 + X(1)^2 - X(6)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[p3] + x[p4] + x[p5] - x[p1] + T(8) * (x[p3]^3 - x[p3] * x[p2]) -
        T(4) * x[p4]^2 + x[p2]^2 - x[p5]^2 + x[p1]^2 - x[p6]^2 - T(2),
    )

    # C(4): 6*X(N-2)+X(N-1)+X(N)-X(N-4)-X(N-5) + 8*(X(N-2)^3-X(N-2)*X(N-3)) - 4*X(N-1)^2 + X(N-3)^2 - X(N)^2 + X(N-4)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[pNm2] + x[pNm1] + x[pN] - x[pNm4] - x[pNm5] +
        T(8) * (x[pNm2]^3 - x[pNm2] * x[pNm3]) - T(4) * x[pNm1]^2 + x[pNm3]^2 -
        x[pN]^2 + x[pNm4]^2 - T(2),
    )

    # C(5): 6*X(N-1)-X(N-3)+X(N)-X(N-4) + 8*(X(N-1)^3-X(N-1)*X(N-2)) - 4*X(N)^2 + X(N-2)^2 + X(N-3)^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[pNm1] - x[pNm3] + x[pN] - x[pNm4] +
        T(8) * (x[pNm1]^3 - x[pNm1] * x[pNm2]) - T(4) * x[pN]^2 + x[pNm2]^2 +
        x[pNm3]^2 - T(2),
    )

    # C(6): 2*X(N)-X(N-3)-X(N-2) + 8*(X(N)^3-X(N)*X(N-1)) + X(N-1)^2 + X(N-2)^2 = 0
    ExaModels.@add_con(
        c,
        T(2) * x[pN] - x[pNm3] - x[pNm2] + T(8) * (x[pN]^3 - x[pN] * x[pNm1]) +
        x[pNm1]^2 + x[pNm2]^2,
    )

    return ExaModels.ExaModel(c; kwargs...)
end

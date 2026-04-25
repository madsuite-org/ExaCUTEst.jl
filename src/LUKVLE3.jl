# LUKVLE3: Chained Powell singular function with simplified trigonometric exponential constraints
# Source: Problem 5.3, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 2 equality constraints
# All variables free; start: X(4k+1)=3, X(4k+2)=-1, X(4k+3)=0, X(4k+4)=1
#
# CUTEst variable ordering (due to sifdecoder reordering):
#   SIF X(3) ↔ CUTEst pos N (swap)
#   SIF X(4) ↔ CUTEst pos N-1 (swap)
#   All other SIF X(k) = CUTEst pos k (identity)
#
# Objective (GROUP TYPES L2P: P*v^2, L4P: P*v^4):
#   sum_{I=1}^{N/2-1} [1*(X(2I-1)+10*X(2I))^2 + 5*(X(2I+1)-X(2I+2))^2
#                      + 1*(X(2I)-2*X(2I+1))^4 + 10*(X(2I-1)-X(2I+2))^4]
#
# Constraints (2 equality constraints, identity type):
#   C1: 3*X(1)^3 + sin(X(1)-X(2))*sin(X(1)+X(2)) + 2*X(2) - 5 = 0
#   C2: 4*X(N-1) - X(N-1)*exp(X(N-1)-X(N)) - 3 = 0
#   (where X(N-1)=CUTEst pos 4 and X(N)=CUTEst pos 3 due to permutation)
function lukvle3_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    @assert N >= 6 "N must be >= 6"
    N2m1 = N ÷ 2 - 1  # = N/2 - 1

    # Build SIF → CUTEst permutation:
    # perm[3] = N, perm[4] = N-1, perm[N-1] = 4, perm[N] = 3, others identity
    perm = collect(1:N)
    perm[3] = N
    perm[4] = N - 1
    perm[N-1] = 4
    perm[N] = 3

    # Start: X(4k+1)=3, X(4k+2)=-1, X(4k+3)=0, X(4k+4)=1 (in SIF order → CUTEst order)
    sif_start = T[if k % 4 == 1; T(3) elseif k % 4 == 2; T(-1) elseif k % 4 == 3; T(0) else T(1) end for k in 1:N]
    start = T[sif_start[invperm(perm)[c]] for c in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: groups OA(I)..OD(I) for I=1..N/2-1
    # Using CUTEst positions p[k] = perm[k]:
    # For most groups (I=1..N/2-2 where 2I+2 < N-1), all SIF vars at natural positions
    # Boundary groups involve perm[3]=N, perm[4]=N-1, perm[N-1]=4, perm[N]=3
    # Group I=1: SIF X(1..4) → CUTEst pos (1, 2, N, N-1)
    # Group I=N/2-1: SIF X(N-3..N) → CUTEst pos (N-3, N-2, 4, 3)
    # All other groups I=2..N/2-2: SIF X(2I-1..2I+2) → CUTEst pos 2I-1..2I+2 (identity)

    # Group I=1: SIF pos (1,2,3,4) → CUTEst (1, 2, perm[3]=N, perm[4]=N-1)
    let pa = perm[1], pb = perm[2], pc = perm[3], pd = perm[4]
        ExaModels.@add_obj(c, (x[pa] + T(10) * x[pb])^2)
        ExaModels.@add_obj(c, T(5) * (x[pc] - x[pd])^2)
        ExaModels.@add_obj(c, (x[pb] - T(2) * x[pc])^4)
        ExaModels.@add_obj(c, T(10) * (x[pa] - x[pd])^4)
    end

    # Interior groups I=2..N/2-2: all natural positions
    if N2m1 >= 3
        ExaModels.@add_obj(c, (x[2i-1] + T(10) * x[2i])^2 for i in 2:(N2m1-1))
        ExaModels.@add_obj(c, T(5) * (x[2i+1] - x[2i+2])^2 for i in 2:(N2m1-1))
        ExaModels.@add_obj(c, (x[2i] - T(2) * x[2i+1])^4 for i in 2:(N2m1-1))
        ExaModels.@add_obj(c, T(10) * (x[2i-1] - x[2i+2])^4 for i in 2:(N2m1-1))
    end

    # Group I=N/2-1: SIF pos (N-3,N-2,N-1,N) → CUTEst (N-3, N-2, perm[N-1]=4, perm[N]=3)
    let pa = perm[N-3], pb = perm[N-2], pc = perm[N-1], pd = perm[N]
        ExaModels.@add_obj(c, (x[pa] + T(10) * x[pb])^2)
        ExaModels.@add_obj(c, T(5) * (x[pc] - x[pd])^2)
        ExaModels.@add_obj(c, (x[pb] - T(2) * x[pc])^4)
        ExaModels.@add_obj(c, T(10) * (x[pa] - x[pd])^4)
    end

    # C1: 3*X(1)^3 + sin(X(1)-X(2))*sin(X(1)+X(2)) + 2*X(2) - 5
    # SIF X(1)=CUTEst pos 1, SIF X(2)=CUTEst pos 2
    ExaModels.@add_con(
        c,
        T(3) * x[1]^3 + sin(x[1] - x[2]) * sin(x[1] + x[2]) + T(2) * x[2] - T(5),
    )

    # C2: 4*X(N-1) - X(N-1)*exp(X(N-1)-X(N)) - 3
    # SIF X(N-1)=CUTEst pos perm[N-1]=4, SIF X(N)=CUTEst pos perm[N]=3
    let pNm1 = perm[N-1], pN = perm[N]
        ExaModels.@add_con(
            c,
            T(4) * x[pNm1] - x[pNm1] * exp(x[pNm1] - x[pN]) - T(3),
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end

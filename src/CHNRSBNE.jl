# CHNRSBNE: Chained Rosenbrock nonlinear equations (Toint), N=50 variables, 98 constraints
# Source: Ph.L. Toint (1978); Buckley #46
# classification NOR2-AN-V-V
#
# Variables: X(1)..X(N), all free, start = -1.0
#
# 98 equality constraints (lcon=ucon=0):
#   B(I) for I=2..N: v_B(I) = X(I) - 1   (linear group, no element)
#   SQ(I) for I=2..N: v_SQ(I) = SCALE_I * (X(I-1) - X(I)^2)
#     where SCALE_I = 4*ALPH(I) and ALPH(I) are given constants
#     SCALE comes from SIF: AM AI ALPH(I) 1.0; RM 4AI AI 4.0; RD SCL 4AI 1.0 → SCL=1/(4AI)
#     constraint value = v / SCL = 4*ALPH(I)*(X(I-1) - X(I)^2)
#
# Cyclic structure (observed in CUTEst): CUTEst's decoder wraps the endpoints:
#   SQ(2)  uses x[N]  as linear part (instead of x[1])
#   SQ(N)  uses x[1]  as element (instead of x[N])
#
# CUTEst constraint ordering follows a butterfly permutation:
#   B constraints in order: B(1), B(2), B(N-1), B(3), B(N-2), ..., B(N/2), B(N/2+1)
#   where B(I) here means "x[I]-1" (CUTEst adds B(1)=x[1]-1 as first constraint)
#   SQ constraints in butterfly from center: SQ(N/2+1), SQ(N/2+2), SQ(N/2), ..., SQ(N), SQ(2)
function chnrsbne_model(; T = Float64, backend = nothing, kwargs...)
    N = 50
    alph = T[
        1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
        1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
        1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
        1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
        2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50,
    ]

    # Butterfly permutation for B constraints (x[j]-1 in this order):
    # [1, 2, N-1, 3, N-2, 4, N-3, ..., N/2, N/2+1]
    b_perm = let
        perm = Int[]
        push!(perm, 1)
        lo = 2; hi = N - 1
        while lo <= hi
            push!(perm, lo)
            if lo != hi
                push!(perm, hi)
            end
            lo += 1; hi -= 1
        end
        perm
    end

    # Butterfly permutation for SQ constraints (I in this order):
    # [N/2+1, N/2+2, N/2, N/2+3, N/2-1, ..., N, 2]
    sq_perm = let
        perm = Int[]
        mid = div(N, 2) + 1  # = 26
        push!(perm, mid)
        for k in 1:(div(N, 2) - 2)
            push!(perm, mid + k, mid - k)
        end
        push!(perm, N, 2)  # cyclic endpoints last
        perm
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N;
        lvar  = fill(T(-Inf), N),
        uvar  = fill(T(Inf), N),
        start = fill(T(-1.0), N))

    # B constraints: x[j] - 1 = 0, ordered by b_perm
    for j in b_perm
        ExaModels.@add_con(c, x[j] - T(1))
    end

    # SQ constraints: 4*ALPH(I)*(x[lin_idx] - x[elem_idx]^2) = 0, ordered by sq_perm
    # Cyclic wrapping: SQ(2) uses x[N] as linear; SQ(N) uses x[1] as element
    for i in sq_perm
        lin_idx  = (i > 2) ? (i - 1) : N
        elem_idx = (i < N) ? i : 1
        ai = T(4) * alph[i]
        ExaModels.@add_con(c, ai * (x[lin_idx] - x[elem_idx]^2))
    end

    return ExaModels.ExaModel(c; kwargs...)
end

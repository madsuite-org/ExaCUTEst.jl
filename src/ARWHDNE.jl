# ARWHDNE: Arrow-head nonlinear equations problem (N variables, 2*(N-1) constraints)
# Zero objective; constraints:
#   L(I) for I=1..N-1: -4*X(I) - (-3) = 3 - 4*X(I) = 0
#   G(I) for I=1..N-1: X(I)^2 + X(N)^2 = 0
# All variables free, starting point x = 1.0
# Default N = 500 (matching SIF default)
#
# SIF structure:
#   NGS = N - 1
#   L(I) groups (E type, identity): linear -4*X(I), constant -3
#     → v = -4*X(I) - (-3) = 3 - 4*X(I)
#   G(I) groups (E type, identity): elements A(I)=X(I)^2, B(I)=X(N)^2
#     → v = X(I)^2 + X(N)^2
#
# CUTEst constraint ordering (verified empirically for default N=500):
#   Rows 1..NGS: all L(I) groups in interleaved order
#     - odd rows:  L(1), L(2), ..., L(mid)
#     - even rows: L(NGS), L(NGS-1), ..., L(mid+1)
#     combined: [1, NGS, 2, NGS-1, ..., mid]
#   Rows NGS+1..2*NGS: all G(I) groups in interleaved order
#     - odd rows:  G(mid), G(mid+1), ..., G(NGS)
#     - even rows: G(mid-1), G(mid-2), ..., G(1)
#     combined: [mid, mid-1, mid+1, mid-2, ..., NGS, 1]
function arwhdne_model(N = 500; T = Float64, backend = nothing, kwargs...)
    NGS = N - 1
    mid = div(NGS + 1, 2)

    # Build constraint ordering permutations (matching CUTEst sifdecoder ordering)
    LP = Vector{Int}(undef, NGS)
    for k in 1:mid
        LP[2k-1] = k
    end
    for k in 1:(NGS - mid)
        LP[2k] = NGS + 1 - k
    end

    GP = Vector{Int}(undef, NGS)
    for k in 1:mid
        GP[2k-1] = mid - 1 + k
    end
    for k in 1:(NGS - mid)
        GP[2k] = mid - k
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    # Zero objective (NOR2 problem)
    # L(I) constraints: 3 - 4*x[LP[k]] = 0  (in CUTEst ordering)
    ExaModels.@add_con(c, T(3) - T(4) * x[i] for i in LP)
    # G(I) constraints: x[GP[k]]^2 + x[N]^2 = 0  (in CUTEst ordering)
    ExaModels.@add_con(c, x[i]^2 + x[N]^2 for i in GP)
    return ExaModels.ExaModel(c; kwargs...)
end

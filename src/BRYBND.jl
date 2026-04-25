# BRYBND: Broyden banded system of nonlinear equations, least-squares sense (unconstrained)
# Source: More', Garbow, Hillstrom (1981), problem 31; Buckley#73; Toint#18
# Classification: SUR2-AN-V-0
# Default N = 5000, all FREE (FR DEFAULT), start = 1.0
# KAPPA1=2.0, KAPPA2=5.0, KAPPA3=1.0, LB=5 (lower bandwidth), UB=1 (upper bandwidth)
#
# ELEMENT TYPES: SQ(V) = V^2,  CB(V) = V^3
# GROUP TYPE L2: F(GVAR) = GVAR^2  (contribution = v^2)
#
# Three band regions (with LB=5, UB=1):
#   Upper left  I=1..LB:     v = K1*x[I] + sum_{j<I} -K3*(x[j]+x[j]^2) + sum_{j in upper} -K3*(x[j]+x[j]^2) + K2*x[I]^3
#   Main        I=LB+1..N-UB-1: v = K1*x[I] + sum_{j<I,LB} -K3*(x[j]+x[j]^3) + sum_{j>I,UB} -K3*(x[j]+x[j]^2) + K2*x[I]^2
#   Lower right I=N-UB..N:   v = K1*x[I] + sum_{j<I,LB} -K3*(x[j]+x[j]^2) + sum_{j>I,N} -K3*(x[j]+x[j]^2) + K2*x[I]^3
#
# Full group value v[I] = linear_part[I] + element_contribution[I]; objective = sum_I v[I]^2
#
# Upper left (I=1..5): lower band = 1..I-1 (grows with I); upper band = I+1..I+UB
#   Elements: SQ for all neighbors (both sides), CB for diagonal
# Main (I=6..N-2): lower band = I-LB..I-1 (fixed width LB=5); upper band = I+1..I+UB
#   Elements: CB for lower-band neighbors, SQ for diagonal and upper-band neighbors
# Lower right (I=N-1..N): lower band = I-LB..I-1; upper band = I+1..N (truncated)
#   Elements: SQ for all neighbors (both sides), CB for diagonal
#
# Note: with LB=5 and UB=1, all upper-left/lower-right boundary groups are enumerated
# explicitly since they have asymmetric band widths.
function brybnd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    # Parameters
    K1 = T(2.0)  # KAPPA1: x[I] linear coefficient
    K2 = T(5.0)  # KAPPA2: CB/SQ element coefficient for diagonal
    K3 = T(1.0)  # KAPPA3: neighbor coefficient (negated in SIF: -KAPPA3)
    LB = 5; UB = 1
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))

    # ── Upper left: I=1..LB (=5) ─────────────────────────────────────────────
    # Lower-band neighbors: j=1..I-1 (growing); upper-band: j=I+1..I+UB=I+1
    # Elements: SQ for neighbors, CB for diagonal
    # v[I] = K1*x[I] + K2*x[I]^3 + sum_{j=1..I-1} (-K3)*(x[j]+x[j]^2) + (-K3)*(x[I+1]+x[I+1]^2)

    # I=1: no lower neighbors; upper = {2}
    ExaModels.@add_obj(
        c,
        (K1 * x[1] + K2 * x[1]^3 - K3 * (x[2] + x[2]^2))^2
    )
    # I=2: lower = {1}; upper = {3}
    ExaModels.@add_obj(
        c,
        (K1 * x[2] + K2 * x[2]^3 - K3 * (x[1] + x[1]^2) - K3 * (x[3] + x[3]^2))^2
    )
    # I=3: lower = {1,2}; upper = {4}
    ExaModels.@add_obj(
        c,
        (K1 * x[3] + K2 * x[3]^3 - K3 * (x[1] + x[1]^2) - K3 * (x[2] + x[2]^2) -
         K3 * (x[4] + x[4]^2))^2
    )
    # I=4: lower = {1,2,3}; upper = {5}
    ExaModels.@add_obj(
        c,
        (K1 * x[4] + K2 * x[4]^3 - K3 * (x[1] + x[1]^2) - K3 * (x[2] + x[2]^2) -
         K3 * (x[3] + x[3]^2) - K3 * (x[5] + x[5]^2))^2
    )
    # I=5: lower = {1,2,3,4}; upper = {6}
    ExaModels.@add_obj(
        c,
        (K1 * x[5] + K2 * x[5]^3 - K3 * (x[1] + x[1]^2) - K3 * (x[2] + x[2]^2) -
         K3 * (x[3] + x[3]^2) - K3 * (x[4] + x[4]^2) - K3 * (x[6] + x[6]^2))^2
    )

    # ── Main: I=LB+1..N-UB-1 (=6..N-2) ──────────────────────────────────────
    # Lower band: j=I-LB..I-1 (LB=5 neighbors); upper band: j=I+1..I+UB=I+1
    # Elements: CB for lower-band neighbors; SQ for diagonal; SQ for upper-band
    # v[I] = K1*x[I] + K2*x[I]^2
    #      + sum_{k=1..5} (-K3)*(x[I-k] + x[I-k]^3)
    #      + (-K3)*(x[I+1] + x[I+1]^2)
    ExaModels.@add_obj(
        c,
        (K1 * x[i] + K2 * x[i]^2
         - K3 * (x[i-1] + x[i-1]^3) - K3 * (x[i-2] + x[i-2]^3) - K3 * (x[i-3] + x[i-3]^3) -
         K3 * (x[i-4] + x[i-4]^3) - K3 * (x[i-5] + x[i-5]^3) - K3 * (x[i+1] + x[i+1]^2))^2
        for i in LB+1:N-UB-1
    )

    # ── Lower right: I=N-UB..N (=N-1..N) ────────────────────────────────────
    # Lower band: j=I-LB..I-1 (LB=5 neighbors, full); upper band truncated to j=I+1..N
    # Elements: SQ for neighbors, CB for diagonal
    # v[I] = K1*x[I] + K2*x[I]^3 + sum_{j=I-LB..I-1} (-K3)*(x[j]+x[j]^2) + sum_{j=I+1..N} (-K3)*(x[j]+x[j]^2)

    # I=N-1: lower = {N-6,N-5,N-4,N-3,N-2}; upper = {N}
    ExaModels.@add_obj(
        c,
        (K1 * x[N-1] + K2 * x[N-1]^3
         - K3 * (x[N-6] + x[N-6]^2) - K3 * (x[N-5] + x[N-5]^2) - K3 * (x[N-4] + x[N-4]^2) -
         K3 * (x[N-3] + x[N-3]^2) - K3 * (x[N-2] + x[N-2]^2) - K3 * (x[N] + x[N]^2))^2
    )
    # I=N: lower = {N-5,N-4,N-3,N-2,N-1}; upper = {} (empty)
    ExaModels.@add_obj(
        c,
        (K1 * x[N] + K2 * x[N]^3
         - K3 * (x[N-5] + x[N-5]^2) - K3 * (x[N-4] + x[N-4]^2) - K3 * (x[N-3] + x[N-3]^2) -
         K3 * (x[N-2] + x[N-2]^2) - K3 * (x[N-1] + x[N-1]^2))^2
    )

    return ExaModels.ExaModel(c; kwargs...)
end

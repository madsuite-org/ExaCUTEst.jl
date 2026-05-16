# CLPLATEC: Clamped plate problem C (P=71 default, unconstrained)
# Source: J. Nocedal (1981), Cocoyoc conference proceedings
# SIF classification: OXR2-MN-V-0
#
# Same variable/bound/start structure as CLPLATEA.
#
# GROUPS (note: SIF has ZN A(I,J) 'SCALE' 1/HP2 overriding initial 2.0 for A):
#   A(I,J): L2 type, scale=1/HP2 (ZN overrides initial 2.0),
#            v = X(I,J) - X(I,J-1) → contribution = v^2 * HP2
#   B(I,J): L2 type, scale=2.0,
#            v = X(I,J) - X(I-1,J) → contribution = v^2 / 2
#   C(I,J): L4 type, no scale (=1),
#            v = X(I,J) - X(I,J-1) → contribution = v^4
#   D(I,J): L4 type, no scale (=1),
#            v = X(I,J) - X(I-1,J) → contribution = v^4
# W group: identity, contribution = WR*X(P,P) + WL*X(P,1)
#   where WR = WGHT*0.99, WL = WGHT*0.01, WGHT = -0.1
function clplatec_model(P = 71; T = Float64, backend = nothing, kwargs...)
    N     = P * P
    HP2   = T(P * P) * T(0.5)      # = P^2 / 2
    WGHT  = T(-0.1)
    WR    = WGHT * T(0.99)
    WL    = WGHT * T(0.01)

    # Variable index: X(I,J) → linear index (J-1)*P + I
    idx(i, j) = (j - 1) * P + i

    # Bounds: all free, except X(1,J) fixed to 0 for J=1..P
    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf),  N)
    for j in 1:P
        k = idx(1, j)
        lvar[k] = T(0)
        uvar[k] = T(0)
    end

    # Enumerate grid pairs for groups (I=2..P, J=2..P)
    grid_pairs = [(i, j) for i in 2:P for j in 2:P]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = T(0))

    # A(I,J): L2 group, scale=1/HP2 (ZN overrides 2.0) → (X(I,J)-X(I,J-1))^2 * HP2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i,j-1)])^2 * HP2
        for (i, j) in grid_pairs
    )

    # B(I,J): L2 group, scale=2.0 → (X(I,J)-X(I-1,J))^2 / 2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i-1,j)])^2 / T(2)
        for (i, j) in grid_pairs
    )

    # C(I,J): L4 group, scale=1 → (X(I,J)-X(I,J-1))^4
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i,j-1)])^4
        for (i, j) in grid_pairs
    )

    # D(I,J): L4 group, scale=1 → (X(I,J)-X(I-1,J))^4
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i-1,j)])^4
        for (i, j) in grid_pairs
    )

    # W group: identity type, contribution = WR*X(P,P) + WL*X(P,1)
    ExaModels.@add_obj(c, WR * x[idx(P, P)] + WL * x[idx(P, 1)])

    return ExaModels.ExaModel(c; kwargs...)
end

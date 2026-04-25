# CLPLATEA: Clamped plate problem (P=71 default, unconstrained)
# Source: J. Nocedal (1981), Cocoyoc conference proceedings
# SIF classification: OXR2-MN-V-0
#
# P*P variables X(I,J), linear index idx(I,J) = (J-1)*P + I
# Bounds: X(1,J) = 0 (fixed) for J=1..P; all other variables free
# Start: 0 everywhere
#
# Parameters:
#   HP2 = P^2 * 0.5   (= 0.5 * RP2 where RP2 = P^2)
#   1/HP2 = 1/HP2
#   WGHT = -0.1
#
# Groups for I=2..P, J=2..P:
#   A(I,J): L2 type, scale 2.0,   v = X(I,J) - X(I,J-1)  → contribution = v^2/2
#   B(I,J): L2 type, scale 2.0,   v = X(I,J) - X(I-1,J)  → contribution = v^2/2
#   C(I,J): L4 type, scale 1/HP2, v = X(I,J) - X(I,J-1)  → contribution = v^4 * HP2
#   D(I,J): L4 type, scale 1/HP2, v = X(I,J) - X(I-1,J)  → contribution = v^4 * HP2
# Group W: identity type (no scale), linear WGHT*X(P,P) → contribution = WGHT * X(P,P)
function clplatea_model(P = 71; T = Float64, backend = nothing, kwargs...)
    N     = P * P
    HP2   = T(P * P) * T(0.5)      # = P^2 / 2
    WGHT  = T(-0.1)

    # Variable index: X(I,J) → linear index (J-1)*P + I
    idx(i, j) = (j - 1) * P + i

    # Bounds: X(1,J) fixed to 0; all others free
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

    # A(I,J): L2 group, scale=2 → (X(I,J)-X(I,J-1))^2 / 2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i,j-1)])^2 / T(2)
        for (i, j) in grid_pairs
    )

    # B(I,J): L2 group, scale=2 → (X(I,J)-X(I-1,J))^2 / 2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i-1,j)])^2 / T(2)
        for (i, j) in grid_pairs
    )

    # C(I,J): L4 group, scale=1/HP2 → (X(I,J)-X(I,J-1))^4 * HP2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i,j-1)])^4 * HP2
        for (i, j) in grid_pairs
    )

    # D(I,J): L4 group, scale=1/HP2 → (X(I,J)-X(I-1,J))^4 * HP2
    ExaModels.@add_obj(
        c,
        (x[idx(i,j)] - x[idx(i-1,j)])^4 * HP2
        for (i, j) in grid_pairs
    )

    # W group: identity, contribution = WGHT * X(P,P)
    ExaModels.@add_obj(c, WGHT * x[idx(P, P)])

    return ExaModels.ExaModel(c; kwargs...)
end

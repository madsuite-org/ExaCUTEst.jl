# POROUS1: Porous medium equation on unit square
# P×P grid, inner points only ((P-2)^2 constraints)
# Variables U(I,J) for I,J=1..P
# Fixed: U(1,J)=1, U(P,J)=0, U(I,P)=1, U(I,1)=0 for interior I
# Starting: U(I,J) = 1 - x1*x2 = 1 - (I-1)*H*(J-1)*H where H=1/(P-1)
# Constraints G(I,J) for I,J=2..P-1 (equality):
#   (1/H^2)*(U(I+1,J)^2 + U(I-1,J)^2 + U(I,J-1)^2 + U(I,J+1)^2 - 4*U(I,J)^2)
#   + (D/2H)*(U(I+1,J)^3 - U(I-1,J)^3) - rhs = 0
# rhs: G(P-1,P-1) has rhs=-50 (point source), others rhs=0
# D=50.0 for POROUS1; D=-50.0 for POROUS2
# SQ element: F=U^2, CB element: F=U^3
function porous1_model(P = 72; T = Float64, backend = nothing, kwargs...)
    D = T(50.0)
    H = T(1.0 / (P - 1))
    inv_H2 = T(1.0) / (H * H)
    D_2H = D / (T(2) * H)

    # Variable layout: U(I,J) = (J-1)*P + I for I=1..P, J=1..P
    idx(i, j) = (j - 1) * P + i
    nvar = P * P

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    # Fix boundary conditions
    for j in 1:P
        lvar[idx(1, j)] = T(1.0); uvar[idx(1, j)] = T(1.0)  # U(1,J)=1
        lvar[idx(P, j)] = T(0.0); uvar[idx(P, j)] = T(0.0)  # U(P,J)=0
    end
    for i in 2:P-1
        lvar[idx(i, P)] = T(1.0); uvar[idx(i, P)] = T(1.0)  # U(I,P)=1
        lvar[idx(i, 1)] = T(0.0); uvar[idx(i, 1)] = T(0.0)  # U(I,1)=0
    end

    # Starting point: U(I,J) = 1 - x1*x2 = 1 - (I-1)*H*(J-1)*H
    start = zeros(T, nvar)
    for i in 1:P
        for j in 1:P
            x1 = T(i - 1) * H
            x2 = T(j - 1) * H
            start[idx(i, j)] = T(1.0) - x1 * x2
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Constraints G(I,J) for I,J=2..P-1
    # General: inv_H2*(U(I+1,J)^2 + U(I-1,J)^2 + U(I,J-1)^2 + U(I,J+1)^2 - 4*U(I,J)^2)
    #         + D_2H*(U(I+1,J)^3 - U(I-1,J)^3) = 0
    # Special: G(P-1,P-1) has rhs = -50 (or the constraint minus rhs)
    # In SIF: CONSTANTS says G(P-1,P-1) = -50 → constraint = expr - (-50) at rhs ... wait:
    # SIF CONSTANTS: "X POROUS1 G(P-1,P-1) -50.0" means rhs = -50
    # So constraint: expr = rhs → expr + 50 = 0 → we write expr - rhs
    # Standard E constraint: value = 0, so lcon=ucon=0
    # expr - (-50) = expr + 50 ... but wait: rhs=-50 means group value = -50
    # So: (group_expr) - (-50) = group_expr + 50 = 0
    # group_expr = inv_H2*(SQ terms) + D_2H*(CB terms)
    # Since constraint is 0: group_expr + 50 = 0 → group_expr = -50
    # But ExaModels convention: we encode (expr - rhs) = 0
    # Here rhs = -50 → (expr - (-50)) = expr + 50 ... that's wrong
    # Actually: SIF G(P-1,P-1) = -50 means the RHS of the group = -50
    # Group value = linear_part + element_contrib - rhs = 0
    # So: element_contrib + linear_part = rhs
    # constraint = element_contrib + linear_part - rhs = 0
    ExaModels.@add_con(c,
        inv_H2 * (x[idx(i+1, j)]^2 + x[idx(i-1, j)]^2 + x[idx(i, j-1)]^2 + x[idx(i, j+1)]^2 - T(4) * x[idx(i, j)]^2) +
        D_2H * (x[idx(i+1, j)]^3 - x[idx(i-1, j)]^3)
        for i in 2:P-1, j in 2:P-1 if !(i == P-1 && j == P-1))

    # Special constraint G(P-1,P-1): rhs = -50
    let i = P - 1, j = P - 1
        ExaModels.@add_con(c,
            inv_H2 * (x[idx(i+1, j)]^2 + x[idx(i-1, j)]^2 + x[idx(i, j-1)]^2 + x[idx(i, j+1)]^2 - T(4) * x[idx(i, j)]^2) +
            D_2H * (x[idx(i+1, j)]^3 - x[idx(i-1, j)]^3) - T(-50.0))
    end

    return ExaModels.ExaModel(c; kwargs...)
end

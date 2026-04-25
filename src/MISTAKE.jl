# MISTAKE: Bilinear objective with unit-circle/disk constraints (Toint 1990)
# "A mistake in writing Hock and Schittkowski problem 108"
# 9 variables: X1..X8 free, X9 ≥ 0; start = 1.0
# Objective: -0.5*X1*X4 + 0.5*X2*X3 - 0.5*X3*X9 + 0.5*X5*X9 - 0.5*X5*X8 + 0.5*X6*X7
# 13 constraints (L×9, G×3, L×1)
function mistake_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 9;
        lvar = T[T(-Inf),T(-Inf),T(-Inf),T(-Inf),T(-Inf),T(-Inf),T(-Inf),T(-Inf),T(0)],
        uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c,
        T(-0.5)*x[1]*x[4] + T(0.5)*x[2]*x[3]
        - T(0.5)*x[3]*x[9] + T(0.5)*x[5]*x[9]
        - T(0.5)*x[5]*x[8] + T(0.5)*x[6]*x[7])
    # C1: X3²+X4² ≤ 1
    ExaModels.@add_con(c, x[3]^2 + x[4]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C2: X5²+X6² ≤ 1
    ExaModels.@add_con(c, x[5]^2 + x[6]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C3: X9² ≤ 1
    ExaModels.@add_con(c, x[9]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C4: X1²+(X2-X9)² ≤ 1
    ExaModels.@add_con(c, x[1]^2 + (x[2]-x[9])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C5: (X1-X5)²+(X2-X6)² ≤ 1
    ExaModels.@add_con(c, (x[1]-x[5])^2 + (x[2]-x[6])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C6: (X1-X7)²+(X2-X8)² ≤ 1
    ExaModels.@add_con(c, (x[1]-x[7])^2 + (x[2]-x[8])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C7: (X3-X5)²+(X4-X6)² ≤ 1
    ExaModels.@add_con(c, (x[3]-x[5])^2 + (x[4]-x[6])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C8: (X3-X7)²+(X4-X8)² ≤ 1
    ExaModels.@add_con(c, (x[3]-x[7])^2 + (x[4]-x[8])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C9: X7²+X8*X9 ≤ 1
    ExaModels.@add_con(c, x[7]^2 + x[8]*x[9] - T(1); lcon = T(-Inf), ucon = T(0))
    # C10: X8*X9 ≥ 0
    ExaModels.@add_con(c, x[8]*x[9]; lcon = T(0), ucon = T(Inf))
    # C11: X5*X8 - X6*X7 ≥ 0
    ExaModels.@add_con(c, x[5]*x[8] - x[6]*x[7]; lcon = T(0), ucon = T(Inf))
    # C12: X1*X4 - X2*X3 ≥ 0
    ExaModels.@add_con(c, x[1]*x[4] - x[2]*x[3]; lcon = T(0), ucon = T(Inf))
    # C13: -X5*X9 ≤ 0
    ExaModels.@add_con(c, -x[5]*x[9]; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end

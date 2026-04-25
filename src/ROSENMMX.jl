# ROSENMMX: Nonlinear minmax in four variables (Rosen/Makela 1990)
# 5 free variables: X1, X2, X3, X4, U; start = 0
# Objective: U; 4 L-type constraints (L type = ≤ 0)
function rosenmmx_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, x[5])
    # F1: -U - 5X1 - 5X2 - 21X3 + 7X4 + X1² + X2² + 2X3² + X4² ≤ 0
    ExaModels.@add_con(c,
        -x[5] - T(5)*x[1] - T(5)*x[2] - T(21)*x[3] + T(7)*x[4]
        + x[1]^2 + x[2]^2 + T(2)*x[3]^2 + x[4]^2;
        lcon = T(-Inf), ucon = T(0))
    # F2: -U + 5X1 - 15X2 - 11X3 - 3X4 + 11X1² + 11X2² + 12X3² + 11X4² - 80 ≤ 0
    ExaModels.@add_con(c,
        -x[5] + T(5)*x[1] - T(15)*x[2] - T(11)*x[3] - T(3)*x[4]
        + T(11)*x[1]^2 + T(11)*x[2]^2 + T(12)*x[3]^2 + T(11)*x[4]^2 - T(80);
        lcon = T(-Inf), ucon = T(0))
    # F3: -U - 15X1 - 5X2 - 21X3 - 3X4 + 11X1² + 21X2² + 12X3² + 21X4² - 100 ≤ 0
    ExaModels.@add_con(c,
        -x[5] - T(15)*x[1] - T(5)*x[2] - T(21)*x[3] - T(3)*x[4]
        + T(11)*x[1]^2 + T(21)*x[2]^2 + T(12)*x[3]^2 + T(21)*x[4]^2 - T(100);
        lcon = T(-Inf), ucon = T(0))
    # F4: -U + 15X1 - 15X2 - 21X3 - 3X4 + 11X1² + 11X2² + 12X3² + X4² - 50 ≤ 0
    ExaModels.@add_con(c,
        -x[5] + T(15)*x[1] - T(15)*x[2] - T(21)*x[3] - T(3)*x[4]
        + T(11)*x[1]^2 + T(11)*x[2]^2 + T(12)*x[3]^2 + x[4]^2 - T(50);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end

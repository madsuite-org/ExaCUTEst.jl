# HS106: Heat exchanger design, 8 variables, 6 constraints
# PROD element type: F = W * X * Y
# Objective: X1 + X2 + X3
# All constraints are G type (≥ 0)
# C1 (G≥0): -0.0025*X4 - 0.0025*X6 + 1  (CONSTANT=-1 → subtract → +1)
# C2 (G≥0): 0.0025*X4 - 0.0025*X5 - 0.0025*X7 + 1
# C3 (G≥0): 0.01*X5 - 0.01*X8 + 1
# C4 (G≥0): -100*X1 - 833.33252*X4 + 83333.333 + X1*X6
# C5 (G≥0): 1250*X4 - 1250*X5 + X2*X7 - X2*X4
# C6 (G≥0): 2500*X5 - 1250000 + X3*X8 - X3*X5
function hs106_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8;
                       lvar  = T[100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       uvar  = T[10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
                       start = T[5000.0, 5000.0, 5000.0, 200.0, 350.0, 150.0, 225.0, 425.0])
    # Objective: X1 + X2 + X3
    ExaModels.@add_obj(c, x[1] + x[2] + x[3])
    # C1 (G≥0): -0.0025*X4 - 0.0025*X6 + 1
    ExaModels.@add_con(c,
        T(-0.0025)*x[4] + T(-0.0025)*x[6] + T(1);
        lcon = T(0), ucon = T(Inf))
    # C2 (G≥0): 0.0025*X4 - 0.0025*X5 - 0.0025*X7 + 1
    ExaModels.@add_con(c,
        T(0.0025)*x[4] + T(-0.0025)*x[5] + T(-0.0025)*x[7] + T(1);
        lcon = T(0), ucon = T(Inf))
    # C3 (G≥0): 0.01*X5 - 0.01*X8 + 1
    ExaModels.@add_con(c,
        T(0.01)*x[5] + T(-0.01)*x[8] + T(1);
        lcon = T(0), ucon = T(Inf))
    # C4 (G≥0): -100*X1 - 833.33252*X4 + 83333.333 + X1*X6
    ExaModels.@add_con(c,
        T(-100)*x[1] + T(-833.33252)*x[4] + T(83333.333) + x[1]*x[6];
        lcon = T(0), ucon = T(Inf))
    # C5 (G≥0): 1250*X4 - 1250*X5 + X2*X7 - X2*X4
    ExaModels.@add_con(c,
        T(1250)*x[4] + T(-1250)*x[5] + x[2]*x[7] - x[2]*x[4];
        lcon = T(0), ucon = T(Inf))
    # C6 (G≥0): 2500*X5 - 1250000 + X3*X8 - X3*X5
    ExaModels.@add_con(c,
        T(2500)*x[5] + T(-1250000) + x[3]*x[8] - x[3]*x[5];
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end

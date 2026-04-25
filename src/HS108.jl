# HS108: 9 variables, 13 constraints
# Element types: SQ(X)=X^2, 2PR(X,Y)=X*Y, ISQ(X,Y)=(X-Y)^2
# All vars FREE except X9 with lvar=0. Start=[1.0 for all].
# Objective: -0.5*X1*X4 + 0.5*X2*X3 - 0.5*X3*X9 + 0.5*X5*X9 - 0.5*X5*X8 + 0.5*X6*X7
# C1  (L≤0): X3^2 + X4^2 - 1
# C2  (L≤0): X5^2 + X6^2 - 1
# C3  (L≤0): X9^2 - 1
# C4  (L≤0): X1^2 + (X2-X9)^2 - 1
# C5  (L≤0): (X1-X5)^2 + (X2-X6)^2 - 1
# C6  (L≤0): (X1-X7)^2 + (X2-X8)^2 - 1
# C7  (L≤0): (X3-X5)^2 + (X4-X6)^2 - 1
# C8  (L≤0): (X3-X7)^2 + (X4-X8)^2 - 1
# C9  (L≤0): X7^2 + (X8-X9)^2 - 1
# C10 (G≥0): X3*X9
# C11 (G≥0): X5*X8 - X6*X7
# C12 (G≥0): X1*X4 - X2*X3
# C13 (L≤0): X5*X9
function hs108_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 9;
                       lvar  = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0],
                       uvar  = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Objective
    ExaModels.@add_obj(c,
        T(-0.5)*x[1]*x[4] + T(0.5)*x[2]*x[3] + T(-0.5)*x[3]*x[9] +
        T(0.5)*x[5]*x[9] + T(-0.5)*x[5]*x[8] + T(0.5)*x[6]*x[7])
    # C1 (L≤0): X3^2 + X4^2 - 1
    ExaModels.@add_con(c, x[3]^2 + x[4]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C2 (L≤0): X5^2 + X6^2 - 1
    ExaModels.@add_con(c, x[5]^2 + x[6]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C3 (L≤0): X9^2 - 1
    ExaModels.@add_con(c, x[9]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C4 (L≤0): X1^2 + (X2-X9)^2 - 1
    ExaModels.@add_con(c, x[1]^2 + (x[2] - x[9])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C5 (L≤0): (X1-X5)^2 + (X2-X6)^2 - 1
    ExaModels.@add_con(c, (x[1] - x[5])^2 + (x[2] - x[6])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C6 (L≤0): (X1-X7)^2 + (X2-X8)^2 - 1
    ExaModels.@add_con(c, (x[1] - x[7])^2 + (x[2] - x[8])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C7 (L≤0): (X3-X5)^2 + (X4-X6)^2 - 1
    ExaModels.@add_con(c, (x[3] - x[5])^2 + (x[4] - x[6])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C8 (L≤0): (X3-X7)^2 + (X4-X8)^2 - 1
    ExaModels.@add_con(c, (x[3] - x[7])^2 + (x[4] - x[8])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C9 (L≤0): X7^2 + (X8-X9)^2 - 1
    ExaModels.@add_con(c, x[7]^2 + (x[8] - x[9])^2 - T(1); lcon = T(-Inf), ucon = T(0))
    # C10 (G≥0): X3*X9
    ExaModels.@add_con(c, x[3]*x[9]; lcon = T(0), ucon = T(Inf))
    # C11 (G≥0): X5*X8 - X6*X7
    ExaModels.@add_con(c, x[5]*x[8] - x[6]*x[7]; lcon = T(0), ucon = T(Inf))
    # C12 (G≥0): X1*X4 - X2*X3
    ExaModels.@add_con(c, x[1]*x[4] - x[2]*x[3]; lcon = T(0), ucon = T(Inf))
    # C13 (L≤0): X5*X9
    ExaModels.@add_con(c, x[5]*x[9]; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end

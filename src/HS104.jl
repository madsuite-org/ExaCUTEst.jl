# HS104: 8 variables, 5 constraints
# PROD element type: F = V1^P1 * V2^P2
# CUTEst variable ordering: [X1, X2, X8, X7, X5, X6, X4, X3]
# i.e., x[1]=X1, x[2]=X2, x[3]=X8, x[4]=X7, x[5]=X5, x[6]=X6, x[7]=X4, x[8]=X3
#
# Objective (identity group type):
#   v_OBJ = -X1 - X2 + 10 + 0.4*X1^0.67*X7^(-0.67) + 0.4*X2^0.67*X8^(-0.67)
#         = -x[1] - x[2] + 10 + 0.4*x[1]^0.67*x[4]^(-0.67) + 0.4*x[2]^0.67*x[3]^(-0.67)
# C1 (L≤0): 0.1*X1 + 5.88e-2*X5*X7 - 1 = 0.1*x[1] + 5.88e-2*x[5]*x[4] - 1
# C2 (L≤0): 0.1*X1+0.1*X2 + 5.88e-2*X6*X8 - 1 = 0.1*x[1]+0.1*x[2] + 5.88e-2*x[6]*x[3] - 1
# C3 (L≤0): 4*X3/X5 + 2*X3^(-0.71)/X5 + 5.88e-2*X3^(-1.3)*X7 - 1
#          = 4*x[8]/x[5] + 2*x[8]^(-0.71)/x[5] + 5.88e-2*x[8]^(-1.3)*x[4] - 1
# C4 (L≤0): 4*X4/X6 + 2*X4^(-0.71)/X6 + 5.88e-2*X4^(-1.3)*X8 - 1
#          = 4*x[7]/x[6] + 2*x[7]^(-0.71)/x[6] + 5.88e-2*x[7]^(-1.3)*x[3] - 1
# C5 (G≥0, RANGE=3.2): -X1-X2+9 + 0.4*X1^0.67*X7^(-0.67) + 0.4*X2^0.67*X8^(-0.67)
#                     = -x[1]-x[2]+9 + 0.4*x[1]^0.67*x[4]^(-0.67) + 0.4*x[2]^0.67*x[3]^(-0.67)
function hs104_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst ordering: x[1]=X1, x[2]=X2, x[3]=X8, x[4]=X7, x[5]=X5, x[6]=X6, x[7]=X4, x[8]=X3
    ExaModels.@add_var(c, x, 8;
                       lvar  = T[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       uvar  = T[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       start = T[6.0, 3.0, 0.5, 1.0, 6.0, 6.0, 0.2, 0.4])
    # Objective
    ExaModels.@add_obj(c,
        -x[1] - x[2] + T(10) +
        T(0.4) * x[1]^T(0.67) * x[4]^T(-0.67) +
        T(0.4) * x[2]^T(0.67) * x[3]^T(-0.67))
    # C1 (L≤0)
    ExaModels.@add_con(c,
        T(0.1)*x[1] + T(5.88e-2)*x[5]*x[4] - T(1);
        lcon = T(-Inf), ucon = T(0))
    # C2 (L≤0)
    ExaModels.@add_con(c,
        T(0.1)*x[1] + T(0.1)*x[2] + T(5.88e-2)*x[6]*x[3] - T(1);
        lcon = T(-Inf), ucon = T(0))
    # C3 (L≤0)
    ExaModels.@add_con(c,
        T(4)*x[8]*x[5]^T(-1) + T(2)*x[8]^T(-0.71)*x[5]^T(-1) + T(5.88e-2)*x[8]^T(-1.3)*x[4] - T(1);
        lcon = T(-Inf), ucon = T(0))
    # C4 (L≤0)
    ExaModels.@add_con(c,
        T(4)*x[7]*x[6]^T(-1) + T(2)*x[7]^T(-0.71)*x[6]^T(-1) + T(5.88e-2)*x[7]^T(-1.3)*x[3] - T(1);
        lcon = T(-Inf), ucon = T(0))
    # C5 (G≥0, RANGE=3.2)
    ExaModels.@add_con(c,
        -x[1] - x[2] + T(9) +
        T(0.4) * x[1]^T(0.67) * x[4]^T(-0.67) +
        T(0.4) * x[2]^T(0.67) * x[3]^T(-0.67);
        lcon = T(0), ucon = T(3.2))
    return ExaModels.ExaModel(c; kwargs...)
end

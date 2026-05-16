# SYNTHES1: Synthesis of processing system 1 (MINLP treated as NLP)
# Source: Duran & Grossmann, Math. Prog. 36 (1986), pp. 307-339, problem 1
# 6 vars: x1,x2,x3 continuous [0,2],[0,2],[0,1]; y1,y2,y3 in [0,1]
# Objective: 5y1+6y2+8y3+10x1-7x3+10 - 18*log(x2+1) - 19.2*log(x1-x2+1)
# CUTEst constraint ordering: L6, L5, L3, L4, N2, N1
function synthes1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # x[1..3] = X(1..3), x[4..6] = Y(1..3)
    ExaModels.@add_var(c, x, 6;
        lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        uvar = T[2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    # Objective: OBJ group, constant rhs=-10 → +10 in formula
    # Elements: LOGX2 coeff=-18: -18*log(x2+1); LOGX1X2 coeff=-19.2: -19.2*log(x1-x2+1)
    ExaModels.@add_obj(c, 5 * x[4] + 6 * x[5] + 8 * x[6] + 10 * x[1] - 7 * x[3] + 10 -
                          18 * log(x[2] + 1) - T(19.2) * log(x[1] - x[2] + 1))
    # Constraint ordering (from CUTEst sifdecoder):
    # L6 (L<=0): y1 + y2 - 1 <= 0 (rhs=1.0 from CONSTANTS)
    ExaModels.@add_con(c, x[4] + x[5] - 1; lcon = T(-Inf), ucon = T(0))
    # L5 (L<=0): -x2 + x1 - 2*y2 <= 0
    ExaModels.@add_con(c, -x[2] + x[1] - 2 * x[5]; lcon = T(-Inf), ucon = T(0))
    # L3 (L<=0): x2 - x1 <= 0
    ExaModels.@add_con(c, x[2] - x[1]; lcon = T(-Inf), ucon = T(0))
    # L4 (L<=0): x2 - 2*y1 <= 0
    ExaModels.@add_con(c, x[2] - 2 * x[4]; lcon = T(-Inf), ucon = T(0))
    # N2 (G>=0): -x3 - 2*y3 + 2 + log(x2+1) + 1.2*log(x1-x2+1) >= 0
    # rhs=-2.0 from CONSTANTS, so group_value = linear + 2 + elements
    ExaModels.@add_con(c, -x[3] - 2 * x[6] + 2 + log(x[2] + 1) + T(1.2) * log(x[1] - x[2] + 1);
                       lcon = T(0), ucon = T(Inf))
    # N1 (G>=0): -0.8*x3 + 0.8*log(x2+1) + 0.96*log(x1-x2+1) >= 0
    ExaModels.@add_con(c, -T(0.8) * x[3] + T(0.8) * log(x[2] + 1) + T(0.96) * log(x[1] - x[2] + 1);
                       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end

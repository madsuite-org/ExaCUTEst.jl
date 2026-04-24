function zecevic2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst's internal ordering maps x[1]→X2(SIF), x[2]→X1(SIF)
    # obj = -2*X1 - 3*X2 + 2*X2² = -3*x[1] - 2*x[2] + 2*x[1]²  (CUTEst ordering)
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[10.0, 10.0],
                       start = T[-0.1, 0.1])
    ExaModels.@add_obj(c, -T(3)*x[1] - T(2)*x[2] + T(2)*x[1]^2)
    # CON1: X1+X2≤2 → x[2]+x[1]≤2 ; CON2: X1+4X2≤4 → x[2]+4x[1]≤4
    ExaModels.@add_con(c, x[1] + x[2] - T(2); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, T(4)*x[1] + x[2] - T(4); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end

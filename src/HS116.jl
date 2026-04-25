function hs116_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 13;
                       lvar = T[0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1, 500.0, 0.1, 1.0, 0.0001, 0.0001],
                       uvar = T[1.0, 1.0, 1.0, 0.1, 0.9, 0.9, 1000.0, 1000.0, 1000.0, 500.0, 150.0, 150.0, 150.0],
                       start = T[0.5, 0.8, 0.9, 0.1, 0.14, 0.5, 489.0, 80.0, 650.0, 450.0, 150.0, 150.0, 150.0])
    # OBJ: X11 + X12 + X13
    ExaModels.@add_obj(c, x[11] + x[12] + x[13])
    # C1: X3 - X2 >= 0
    ExaModels.@add_con(c, x[3] - x[2]; lcon=T(0), ucon=T(Inf))
    # C2: X2 - X1 >= 0
    ExaModels.@add_con(c, x[2] - x[1]; lcon=T(0), ucon=T(Inf))
    # C3: -0.002*X7 + 0.002*X8 - (-1) = -0.002*X7 + 0.002*X8 + 1 >= 0
    ExaModels.@add_con(c, -T(0.002)*x[7] + T(0.002)*x[8] + T(1); lcon=T(0), ucon=T(Inf))
    # C4: X11 + X12 + X13 - 50 ∈ [0, 200] (RANGE 200)
    ExaModels.@add_con(c, x[11] + x[12] + x[13] - T(50); lcon=T(0), ucon=T(200))
    # C5: X13 - 1.262626*X10 + 1.231059*X3*X10 >= 0
    ExaModels.@add_con(c, x[13] - T(1.262626)*x[10] + T(1.231059)*x[3]*x[10]; lcon=T(0), ucon=T(Inf))
    # C6: X5 - 0.03475*X2 - 0.975*X2*X5 + 0.00975*X2^2 >= 0
    ExaModels.@add_con(c, x[5] - T(0.03475)*x[2] - T(0.975)*x[2]*x[5] + T(0.00975)*x[2]^2; lcon=T(0), ucon=T(Inf))
    # C7: X6 - 0.03475*X3 - 0.975*X3*X6 + 0.00975*X3^2 >= 0
    ExaModels.@add_con(c, x[6] - T(0.03475)*x[3] - T(0.975)*x[3]*x[6] + T(0.00975)*x[3]^2; lcon=T(0), ucon=T(Inf))
    # C8: X5*X7 - X1*X8 - X4*X7 + X4*X8 >= 0
    ExaModels.@add_con(c, x[5]*x[7] - x[1]*x[8] - x[4]*x[7] + x[4]*x[8]; lcon=T(0), ucon=T(Inf))
    # C9: -X5 - X6 - (-1) - 0.002*X2*X9 - 0.002*X5*X8 + 0.002*X1*X8 + 0.002*X6*X9
    #   = -X5 - X6 + 1 - 0.002*X2*X9 - 0.002*X5*X8 + 0.002*X1*X8 + 0.002*X6*X9 >= 0
    ExaModels.@add_con(c,
        -x[5] - x[6] + T(1) - T(0.002)*x[2]*x[9] - T(0.002)*x[5]*x[8] + T(0.002)*x[1]*x[8] + T(0.002)*x[6]*x[9];
        lcon=T(0), ucon=T(Inf))
    # C10: -500*X2 + 500*X6 + X2*X9 - X3*X10 - X6*X9 + X2*X10 >= 0
    ExaModels.@add_con(c,
        -T(500)*x[2] + T(500)*x[6] + x[2]*x[9] - x[3]*x[10] - x[6]*x[9] + x[2]*x[10];
        lcon=T(0), ucon=T(Inf))
    # C11: X2 - 0.9 - 0.002*X2*X10 + 0.002*X3*X10 >= 0
    ExaModels.@add_con(c, x[2] - T(0.9) - T(0.002)*x[2]*x[10] + T(0.002)*x[3]*x[10]; lcon=T(0), ucon=T(Inf))
    # C12: X4 - 0.03475*X1 - 0.975*X1*X4 + 0.00975*X1^2 >= 0
    ExaModels.@add_con(c, x[4] - T(0.03475)*x[1] - T(0.975)*x[1]*x[4] + T(0.00975)*x[1]^2; lcon=T(0), ucon=T(Inf))
    # C13: X11 - 1.262626*X8 + 1.231059*X1*X8 >= 0
    ExaModels.@add_con(c, x[11] - T(1.262626)*x[8] + T(1.231059)*x[1]*x[8]; lcon=T(0), ucon=T(Inf))
    # C14: X12 - 1.262626*X9 + 1.231059*X2*X9 >= 0
    ExaModels.@add_con(c, x[12] - T(1.262626)*x[9] + T(1.231059)*x[2]*x[9]; lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end

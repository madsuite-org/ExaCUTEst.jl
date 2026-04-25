function palmer4c_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6, A8, A10, A12, A14 (all free); start = 1.0
    ExaModels.@add_var(c, x, 8;
        lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M = 23
    ExaModels.@add_par(c, XI, T[
        -1.658063, -1.570796, -1.396263, -1.221730, -1.047198,
        -0.872665, -0.741119, -0.698132, -0.523599, -0.349066,
        -0.174533,  0.0,
         0.174533,  0.349066,  0.523599,  0.698132,  0.741119,
         0.872665,  1.047198,  1.221730,  1.396263,  1.570796,
         1.658063])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        67.27625, 52.8537,  30.2718,  14.9888,   5.5675,
         0.92603,  0.0,      0.085108,  1.867422,  5.014768,
         8.263520, 9.8046208,
         8.263520, 5.014768,  1.867422,  0.085108,  0.0,
         0.92603,  5.5675,  14.9888,  30.2718,  52.8537,
        67.27625])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + A8*X^8 + A10*X^10 + A12*X^12 + A14*X^14
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]*XI[i]^8 +
         x[6]*XI[i]^10 + x[7]*XI[i]^12 + x[8]*XI[i]^14 - YI[i])^2
        for i in 1:23)
    return ExaModels.ExaModel(c; kwargs...)
end

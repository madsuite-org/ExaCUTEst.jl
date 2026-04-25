function palmer3a_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6 (free), B >= 1e-5, C >= 1e-5; start = 1.0
    ExaModels.@add_var(c, x, 6;
        lvar = T[-Inf, -Inf, -Inf, -Inf, 1e-5, 1e-5],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M = 23
    ExaModels.@add_par(c, XI, T[
        -1.658063, -1.570796, -1.396263, -1.221730, -1.047198,
        -0.872665, -0.766531, -0.698132, -0.523599, -0.349066,
        -0.174533,  0.0,
         0.174533,  0.349066,  0.523599,  0.698132,  0.766531,
         0.872665,  1.047198,  1.221730,  1.396263,  1.570796,
         1.658063])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        64.87939, 50.46046, 28.2034,  13.4575,   4.6547,
         0.59447,  0.0000,   0.2177,   2.3029,   5.5191,
         8.5519,   9.8919,
         8.5519,   5.5191,   2.3029,   0.2177,   0.0000,
         0.59447,  4.6547,  13.4575,  28.2034,  50.46046,
        64.87939])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]/(x[6] + XI[i]^2) - YI[i])^2
        for i in 1:23)
    return ExaModels.ExaModel(c; kwargs...)
end

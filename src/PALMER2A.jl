function palmer2a_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6 (free), B >= 1e-5, C >= 1e-5; start = 1.0
    ExaModels.@add_var(c, x, 6;
        lvar = T[-Inf, -Inf, -Inf, -Inf, 1e-5, 1e-5],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M = 23
    ExaModels.@add_par(c, XI, T[
        -1.745329, -1.570796, -1.396263, -1.221730, -1.047198,
        -0.937187, -0.872665, -0.698132, -0.523599, -0.349066,
        -0.174533,  0.0,
         0.174533,  0.349066,  0.523599,  0.698132,  0.872665,
         0.937187,  1.047198,  1.221730,  1.396263,  1.570796,
         1.745329])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        72.676767, 40.149455, 18.8548,   6.4762,    0.8596,
         0.00000,   0.2730,    3.2043,    8.1080,   13.4291,
        17.7149,   19.4529,
        17.7149,   13.4291,    8.1080,    3.2053,    0.2730,
         0.00000,   0.8596,    6.4762,   18.8548,   40.149455,
        72.676767])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]/(x[6] + XI[i]^2) - YI[i])^2
        for i in 1:23)
    return ExaModels.ExaModel(c; kwargs...)
end

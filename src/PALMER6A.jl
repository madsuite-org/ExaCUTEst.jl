function palmer6a_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6 (free), B >= 1e-5, C >= 1e-5; start = 1.0
    ExaModels.@add_var(c, x, 6;
        lvar = T[-Inf, -Inf, -Inf, -Inf, 1e-5, 1e-5],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M=24, loop from index 12 to 24 (13 points)
    ExaModels.@add_par(c, XI, T[
        0.000000, 1.570796, 1.396263, 1.221730, 1.047198, 0.872665,
        0.785398, 0.732789, 0.698132, 0.610865, 0.523599, 0.349066,
        0.174533])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        10.678659, 75.414511, 41.513459, 20.104735,  7.432436,  1.298082,
         0.171300,  0.000000,  0.068203,  0.774499,  2.070002,  5.574556,
         9.026378])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]/(x[6] + XI[i]^2) - YI[i])^2
        for i in 1:13)
    return ExaModels.ExaModel(c; kwargs...)
end

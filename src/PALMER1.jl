function palmer1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A (free), B >= 1e-5, C >= 1e-5, D >= 1e-5; start = 1.0
    ExaModels.@add_var(c, x, 4;
        lvar = T[-Inf, 1e-5, 1e-5, 1e-5],
        uvar = T[Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0])
    # X data (radians), M = 31
    ExaModels.@add_par(c, XI, T[
        -1.788963, -1.745329, -1.658063, -1.570796, -1.483530,
        -1.396263, -1.308997, -1.218612, -1.134464, -1.047198,
        -0.872665, -0.698132, -0.523599, -0.349066, -0.174533,
         0.0,
         1.788963,  1.745329,  1.658063,  1.570796,  1.483530,
         1.396263,  1.308997,  1.218612,  1.134464,  1.047198,
         0.872665,  0.698132,  0.523599,  0.349066,  0.174533])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        78.596218, 65.77963, 43.96947, 27.038816, 14.6126,
         6.2614,    1.538330,  0.000000,  1.188045,  4.6841,
        16.9321,   33.6988,   52.3664,   70.1630,   83.4221,
        88.3995,
        78.596218, 65.77963, 43.96947, 27.038816, 14.6126,
         6.2614,    1.538330,  0.000000,  1.188045,  4.6841,
        16.9321,   33.6988,   52.3664,   70.1630,   83.4221])
    # Objective: sum of (A*Xi^2 + B/(C + Xi^2/D) - Yi)^2 for i in 1:31
    # L2 group type: contribution = v^2 (no scale factor)
    ExaModels.@add_obj(c,
        (x[1]*XI[i]^2 + x[2]/(x[3] + XI[i]^2/x[4]) - YI[i])^2
        for i in 1:31)
    return ExaModels.ExaModel(c; kwargs...)
end

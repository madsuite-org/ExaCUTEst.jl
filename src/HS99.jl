function hs99_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58],
                       start = T[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ExaModels.@add_obj(c,
        -(T(1250.0)*cos(x[1]) + T(1250.0)*cos(x[2]) + T(3750.0)*cos(x[3]) +
          T(3750.0)*cos(x[4]) + T(3750.0)*cos(x[5]) + T(9000.0)*cos(x[6]) +
          T(9000.0)*cos(x[7]))^2)
    ExaModels.@add_con(c,
        T(459375.0)*sin(x[1]) + T(428125.0)*sin(x[2]) + T(1143750.0)*sin(x[3]) +
        T(956250.0)*sin(x[4]) + T(768750.0)*sin(x[5]) + T(1215000.0)*sin(x[6]) +
        T(405000.0)*sin(x[7]) + T(-2310400.0) - T(100000.0))
    ExaModels.@add_con(c,
        T(1250.0)*sin(x[1]) + T(1250.0)*sin(x[2]) + T(3750.0)*sin(x[3]) +
        T(3750.0)*sin(x[4]) + T(3750.0)*sin(x[5]) + T(9000.0)*sin(x[6]) +
        T(9000.0)*sin(x[7]) + T(-12160.0) - T(1000.0))
    return ExaModels.ExaModel(c; kwargs...)
end

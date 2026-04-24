function hs60_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[-10.0, -10.0, -10.0],
                       uvar = T[10.0, 10.0, 10.0],
                       start = T[2.0, 2.0, 2.0])
    ExaModels.@add_obj(c, (x[1]-1)^2 + (x[1]-x[2])^2 + (x[2]-x[3])^4)
    ExaModels.@add_con(c, x[1] + x[3]^4 + x[1]*x[2]^2 - T(8.242640687))
    return ExaModels.ExaModel(c; kwargs...)
end

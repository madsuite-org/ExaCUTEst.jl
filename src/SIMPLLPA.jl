# SIMPLLPA: Simple LP in 2 variables
# min  2*x1 + x2
# s.t. x1 + x2 >= 1
#      x1 + 2*x2 >= 1.5
# Default bounds: x1 >= 0, x2 >= 0
function simpllpa_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[0.1, 0.1])
    ExaModels.@add_obj(c, T(2) * x[1] + x[2])
    ExaModels.@add_con(c, x[1] + x[2] - T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + T(2) * x[2] - T(1.5); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end

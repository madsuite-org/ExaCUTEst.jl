# EXTROSNBNE: Extended Rosenbrock (nonseparable), nonlinear equations version
# N variables, N equality constraints
# SQ1:    x[1] - 1 = 0
# SQ(I):  (x[i] - x[i-1]^2) / 0.1 = 0   for i=2..N
# All variables free; start: x[i] = -1.0
function extrosnbne_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; start = T(-1.0))
    ExaModels.@add_con(c, x[1] - T(1))
    ExaModels.@add_con(c, (x[i] - x[i-1]^2) / T(0.1) for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end

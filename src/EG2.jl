# EG2: LANCELOT manual example 1.2.4 (Conn et al. 1992)
# N=1000 free variables, no constraints
# Objective = sum_{I=1..N-1} sin(X(1) + X(I)^2 - 1) + 0.5*sin(X(N)^2)
# Start: all 0
function eg2_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, sin(x[1] + x[i]^2 - T(1)) for i in 1:N-1)
    ExaModels.@add_obj(c, T(0.5) * sin(x[N]^2))
    return ExaModels.ExaModel(c; kwargs...)
end

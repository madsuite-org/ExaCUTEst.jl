# MARATOSB: Maratos problem with penalty parameter 1e-6 (2 free vars, start=[1.1,0.1])
# F group (identity, no scale): v = X1 → contribution = X1
# C group (L2, scale=INVP=1e-6): v = X1^2+X2^2-1 → contribution = v^2/1e-6 = 1e6*(X1^2+X2^2-1)^2
function maratosb_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[T(1.1), T(0.1)])
    ExaModels.@add_obj(c, x[1])
    ExaModels.@add_obj(c, T(1e6) * (x[1]^2 + x[2]^2 - T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end

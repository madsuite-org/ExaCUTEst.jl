# MDHOLE: 2D minimization with sine-based objective (Toint 1991)
# 2 variables: X ≥ 0, Y free; start = [10, 1]
# Objective: 100*(sin(X) - Y)² + X
# G1 (L2 group type, SCALE=0.01): v = -Y + sin(X) → contribution = v²/σ = 100*v²
# G2 (identity): v = X → contribution = X
function mdhole_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[T(0), T(-Inf)], uvar = T(Inf),
        start = T[T(10), T(1)])
    ExaModels.@add_obj(c, T(100)*(sin(x[1]) - x[2])^2)
    ExaModels.@add_obj(c, x[1])
    return ExaModels.ExaModel(c; kwargs...)
end

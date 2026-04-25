# HS3MOD: HS3 modification with unit-scale L2 group (Hock & Schittkowski 1981)
# 2 variables: X1 free, X2 ≥ 0; start = [10, 1]
# Objective: X2 + (X2 - X1)²
# GROUP TYPE L2 on G2 (no scale σ=1): contribution = v²/σ = v² where v = -X1 + X2
function hs3mod_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[T(-Inf), T(0)], uvar = T(Inf),
        start = T[T(10), T(1)])
    ExaModels.@add_obj(c, x[2])
    ExaModels.@add_obj(c, (x[2] - x[1])^2)
    return ExaModels.ExaModel(c; kwargs...)
end

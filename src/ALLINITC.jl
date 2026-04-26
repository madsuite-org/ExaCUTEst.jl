# ALLINITC: "All in it" with one equality constraint (Gould 1990)
# Same as ALLINIT + constraint C1 (TRIVIAL type): X1² + X2² = 1
function allinitc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar = T[T(-Inf), T(1), T(-1e10), T(2)],
        uvar = T[T(Inf), T(Inf), T(1), T(2)],
        start = T[T(0), T(0), T(0), T(0)])
    # Objective (same as ALLINIT)
    ExaModels.@add_obj(c, x[3] - T(1))
    ExaModels.@add_obj(c, x[1]^2)
    ExaModels.@add_obj(c, x[2]^2 + (x[3]+x[4])^2)
    ExaModels.@add_obj(c, x[4] + sin(x[3])^2 + x[1]^2*x[2]^2 - T(3))
    ExaModels.@add_obj(c, sin(x[3])^2)
    ExaModels.@add_obj(c, (x[4] - T(1))^2)
    ExaModels.@add_obj(c, x[2]^4)
    ExaModels.@add_obj(c, (x[3]^2 + (x[4]+x[1])^2)^2)
    ExaModels.@add_obj(c, (x[1] + sin(x[4])^2 + x[2]^2*x[3]^2 - T(4))^2)
    ExaModels.@add_obj(c, sin(x[4])^4)
    # C1 (E, TRIVIAL, rhs=1): X1² + X2² = 1
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end

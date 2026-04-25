# EXTRASIM: Extremely simple linear program (Conn et al. 1992)
# 2 variables: X (lvar=0), Y (free); start = [0, 0]
# Objective: X + 1 (from X + constant RHS -(-1.0) = X+1)
# 1 equality constraint: X + 2*Y - 2 = 0
function extrasim_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2;
        lvar = T[T(0), T(-Inf)],
        uvar = T[T(Inf), T(Inf)],
        start = T[T(0), T(0)])
    ExaModels.@add_obj(c, x[1] + T(1))
    ExaModels.@add_con(c, x[1] + T(2)*x[2] - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end

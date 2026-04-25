# RECIPE: Root-finding system (Buckley 1989, problem 155)
# 3 free variables: X1, X2, X3; start = [2, 5, 1]
# 3 equality constraints (root-finding)
# G1: X1 - 5 = 0
# G2: X2² = 0
# G3: X3 / (X2 - X1) = 0   (XOVERU element: V=X3, U=X2-X1, F=V/U)
function recipe_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(2), T(5), T(1)])
    ExaModels.@add_con(c, x[1] - T(5))
    ExaModels.@add_con(c, x[2]^2)
    ExaModels.@add_con(c, x[3] / (x[2] - x[1]))
    return ExaModels.ExaModel(c; kwargs...)
end

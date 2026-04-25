# CLUSTER: Buckley (1989) problem 207, root-finding
# 2 free variables: X1, X2; start = [0, 0]
# No objective; 2 equality constraints:
#   G1: (X1 - X2^2) * (X1 - sin(X2)) = 0
#   G2: (cos(X2) - X1) * (X2 - cos(X1)) = 0
function cluster_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_con(c, (x[1] - x[2]^2) * (x[1] - sin(x[2])))
    ExaModels.@add_con(c, (cos(x[2]) - x[1]) * (x[2] - cos(x[1])))
    return ExaModels.ExaModel(c; kwargs...)
end

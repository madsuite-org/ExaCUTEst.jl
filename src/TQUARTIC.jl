# TQUARTIC: quartic function with repetitious groups (N=5000 variables, unconstrained)
# minimize (x[1] - 1)^2 + sum_{i=2}^{N} (x[1]^2 - x[i]^2)^2
# Source: Ph. Toint, Dec 1989
# SIF: G1: linear x1, constant 1.0, L2 group type → (x1 - 1)^2
#      G(I) for I=2..N: elements E(I)*(-1.0) + E(1)*1.0, L2 group type
#        element SQ(V1) = V1^2; so v = -X(I)^2 + X(1)^2; contribution = v^2
# Default N = 5000 (active IE N line)
function tquartic_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.1))
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_obj(c, (x[1]^2 - x[i]^2)^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end

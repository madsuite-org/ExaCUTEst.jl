# LIARWHD: Simplified nondiagonal problem (N=5000, all free, start=4.0)
# Objective: sum_{I=1}^N [4*(X(I)^2 - X(1))^2 + (X(I) - 1)^2]
# A(I) groups: L2, scale=0.25, linear -X(1), element SQ(X(I))=X(I)^2
#   v = X(I)^2 - X(1), contribution = v^2 / 0.25 = 4*(X(I)^2 - X(1))^2
# B(I) groups: L2, no scale, linear X(I), RHS=1
#   v = X(I) - 1, contribution = (X(I) - 1)^2
function liarwhd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(4))
    ExaModels.@add_obj(c, T(4) * (x[i]^2 - x[1])^2 for i in 1:N)
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end

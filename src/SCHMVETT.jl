# SCHMVETT: Schmidt and Vetters problem (scalable, unconstrained)
# N variables (default N=5000), all free, start = 0.5
# Objective: sum_{i=1}^{N-2} [A(i) + B(i) + C(i)]
#   where:
#   A(i) = SCH1(X(i), X(i+1)): U = X(i) - X(i+1), F = -1/(1 + U^2)
#   B(i) = SCH2(X(i+1), X(i+2)): U = pi*X(i+1) + X(i+2), F = -sin(U/2)
#   C(i) = SCH3(X(i), X(i+1), X(i+2)):
#     U1 = X(i) + X(i+2), U2 = X(i+1),
#     A = U1/U2 - 2, F = -exp(-A^2)
# Groups G(I) have identity type (no GROUP TYPE), contribution = sum of elements
# Source: Schmidt & Vetters (1970), Toint#35, Buckley#14
function schmvett_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    # SCH1 element: U = X(i) - X(i+1), F = -1/(1 + U^2)
    ExaModels.@add_obj(
        c,
        -(one(T) / (one(T) + (x[i] - x[i+1])^2)) for i in 1:N-2
    )
    # SCH2 element: U = pi*X(i+1) + X(i+2), F = -sin(U/2)
    ExaModels.@add_obj(
        c,
        -sin((T(pi) * x[i+1] + x[i+2]) / T(2)) for i in 1:N-2
    )
    # SCH3 element: U1 = X(i) + X(i+2), U2 = X(i+1), A = U1/U2 - 2, F = -exp(-A^2)
    ExaModels.@add_obj(
        c,
        -exp(-((x[i] + x[i+2]) / x[i+1] - T(2))^2) for i in 1:N-2
    )
    return ExaModels.ExaModel(c; kwargs...)
end

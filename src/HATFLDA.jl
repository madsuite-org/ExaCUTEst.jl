# HATFLDA: Hatfield polytechnic test problem (4 variables, unconstrained)
# minimize sum of L2 groups:
#   G(1): v = X(1) - 1.0,            contribution = (X(1) - 1.0)^2
#   G(I) for I=2..4: element SQR(X(I)) = sqrt(X(I)), coeff -1
#     v = X(I-1) - sqrt(X(I)),        contribution = (X(I-1) - sqrt(X(I)))^2
# BOUNDS: lvar = 1e-7 (default XL)
function hatflda_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(1e-7), uvar = T(Inf), start = T(0.1))

    # G(1): v = X(1) - 1.0
    ExaModels.@add_obj(c, (x[1] - T(1))^2)

    # G(I) for I=2..4: v = X(I-1) - sqrt(X(I)), contribution = v^2
    ExaModels.@add_obj(c, (x[i-1] - sqrt(x[i]))^2 for i in 2:4)

    return ExaModels.ExaModel(c; kwargs...)
end

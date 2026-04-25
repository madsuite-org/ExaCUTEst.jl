# CAMEL6: Six-hump camel back function (2 variables, unconstrained)
# minimize 4*X1^2 - 2.1*X1^4 + (1/3)*X1^6 + X1*X2 - 4*X2^2 + 4*X2^4
# Source: L.C.W. Dixon and G.P. Szego (eds.), Towards Global Optimization, North Holland, 1975.
# SIF: N group OBJ (identity), elements SQ, FORPW, SIXPW, 2PROD with coefficients
function camel6_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-3.0, -1.5], uvar = T[3.0, 1.5], start = T[1.1, 1.1])
    ExaModels.@add_obj(
        c,
        T(4.0) * x[1]^2 +
        T(-2.1) * x[1]^4 +
        T(1.0/3.0) * x[1]^6 +
        x[1] * x[2] +
        T(-4.0) * x[2]^2 +
        T(4.0) * x[2]^4
    )
    return ExaModels.ExaModel(c; kwargs...)
end

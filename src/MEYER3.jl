# MEYER3: Meyer thermistor fitting (3 free vars, 16 observations)
# GAUSS element: F(V1,V2,V3;T) = V1*exp(V2/(T+V3))
# G(I): L2, v = GAUSS(X1,X2,X3;T_I) - Y_I  → contribution = v^2
# T_I = 5*I+45 for I=1..16, Y data given as constants
# Variable scales in SIF are just hints; formulation uses actual variable values
function meyer3_model(; T = Float64, backend = nothing, kwargs...)
    Y = T[34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0,
           8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0]
    TP = T[T(5 * i + 45) for i in 1:16]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T[T(0.02), T(4000.0), T(250.0)])
    ExaModels.@add_par(c, YI, Y)
    ExaModels.@add_par(c, TI, TP)
    ExaModels.@add_obj(c, (x[1] * exp(x[2] / (TI[i] + x[3])) - YI[i])^2 for i in 1:16)
    return ExaModels.ExaModel(c; kwargs...)
end

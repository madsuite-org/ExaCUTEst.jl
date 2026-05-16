# S308NE: Nonlinear equations version of S308 (2 variables, 3 equality constraints)
# Source: Schittkowski "More Test Examples for NLP Codes", problem 308
# O1: x1^2 + x1*x2 + x2^2 = 0
# O2: sin(x1) = 0
# O3: cos(x2) = 0
# All variables free; start: x1=3, x2=0.1
function s308ne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
                       start = T[3.0, 0.1])
    # O1: QUAD element = x1^2 + x1*x2 + x2^2
    ExaModels.@add_con(c, x[1]^2 + x[1] * x[2] + x[2]^2)
    # O2: SINE element = sin(x1)
    ExaModels.@add_con(c, sin(x[1]))
    # O3: COSN element = cos(x2)
    ExaModels.@add_con(c, cos(x[2]))
    return ExaModels.ExaModel(c; kwargs...)
end

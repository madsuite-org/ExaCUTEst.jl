# S308: Least-squares problem in 2 variables (unconstrained)
# Source: Schittkowski "More Test Examples for NLP Codes", problem 308
# Groups O1, O2, O3 each use L2 group type (phi=v^2)
# O1: element QUAD(x1,x2): F = x1^2 + x1*x2 + x2^2;  O1 value = F(x1,x2)
# O2: element SINE(x1): F = sin(x1);  O2 value = sin(x1)
# O3: element COSN(x2): F = cos(x2);  O3 value = cos(x2)
# All are L2 groups: obj += v^2
# Total obj = (x1^2 + x1*x2 + x2^2)^2 + sin(x1)^2 + cos(x2)^2
# Bounds: FR 'DEFAULT' -> all free
# Start: x1=3, x2=0.1
function s308_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
                       start = T[3.0, 0.1])
    # O1: L2, v = x1^2 + x1*x2 + x2^2 -> v^2
    ExaModels.@add_obj(c, (x[1]^2 + x[1] * x[2] + x[2]^2)^2)
    # O2: L2, v = sin(x1) -> sin(x1)^2
    ExaModels.@add_obj(c, sin(x[1])^2)
    # O3: L2, v = cos(x2) -> cos(x2)^2
    ExaModels.@add_obj(c, cos(x[2])^2)
    return ExaModels.ExaModel(c; kwargs...)
end

# MADSEN: nonlinear minmax problem (3 variables, 6 G-type inequality constraints)
# Variables: X1=x[1], X2=x[2], U=x[3]; all free; start=[3, 1, 1]
# Objective (N, identity): U
# 6 G-type (>=0) constraints (lcon=0, ucon=Inf):
#   C1: U - X1^2 - X2^2 - X1*X2 >= 0
#   C2: U + X1^2 + X2^2 + X1*X2 >= 0
#   C3: U - sin(X1) >= 0
#   C4: U + sin(X1) >= 0
#   C5: U - cos(X2) >= 0
#   C6: U + cos(X2) >= 0
function madsen_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[3.0, 1.0, 1.0])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, x[3] - x[1]^2 - x[2]^2 - x[1]*x[2]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] + x[1]^2 + x[2]^2 + x[1]*x[2]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - sin(x[1]); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] + sin(x[1]); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - cos(x[2]); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] + cos(x[2]); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end

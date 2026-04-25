# MEXHAT: Mexican hat problem (2 variables, unconstrained)
# Penalty parameter INVP = 0.00001
# Source: Brown and Bartholomew-Biggs (1987)
# SIF:
#   Group F (N, identity): elements O1*(-1) + O2*(-1), no CONSTANT
#     O1 = O2 = SSQ(X=X1, S=1.0): F = (X1-1)^2
#     → contribution = -2*(X1-1)^2
#   Group C (N, L2, SCALE=INVP=0.00001, CONSTANT=0.02):
#     elements C1*10000 + O1*1
#     C1 = XMYSQ(X=X2, Y=X1): F = (X2 - X1^2)^2
#     O1 = SSQ(X=X1, S=1.0): F = (X1-1)^2
#     v_C = 10000*(X2-X1^2)^2 + (X1-1)^2 - 0.02
#     contribution = v_C^2 / 0.00001
#   BOUNDS: FR DEFAULT (free variables)
function mexhat_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[0.86, 0.72])
    # Group F (identity): contribution = -2*(x1-1)^2
    ExaModels.@add_obj(c, -T(2) * (x[1] - T(1))^2)
    # Group C (L2, scale=0.00001): v = 10000*(x2-x1^2)^2 + (x1-1)^2 - 0.02
    #   contribution = v^2 / 0.00001
    ExaModels.@add_obj(c, (T(10000) * (x[2] - x[1]^2)^2 + (x[1] - T(1))^2 - T(0.02))^2 / T(0.00001))
    return ExaModels.ExaModel(c; kwargs...)
end

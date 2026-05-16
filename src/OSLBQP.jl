# OSLBQP: Simple convex QP from OSL manual
# Minimize x1 + 2*x5 - x8 + 0.5*(x1^2+...+x8^2)
# SQ element: F = V1*V1 (so 0.5*SQ = 0.5*V1^2)
# Bounds: x1>=2.5, 0<=x2<=4.1, x3>=0, x4>=0,
#         0.5<=x5<=4.0, x6>=0, x7>=0, 0<=x8<=4.3
function oslbqp_model(; T = Float64, backend = nothing, kwargs...)
    lvar = T[2.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    uvar = T[Inf, 4.1, Inf, Inf, 4.0, Inf, Inf, 4.3]
    start = fill(T(0.5), 8)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = lvar, uvar = uvar, start = start)

    # Linear part: x1 + 2*x5 - x8
    ExaModels.@add_obj(c, x[1] + T(2) * x[5] - x[8])

    # Quadratic part: 0.5*(x1^2+...+x8^2)
    # SQ element: F=V1*V1, coeff=0.5
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:8)

    return ExaModels.ExaModel(c; kwargs...)
end

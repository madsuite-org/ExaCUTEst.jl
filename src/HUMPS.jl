# HUMPS: 2-variable function with many humps
# Source: Ph. Toint, private communication, 1997
#
# obj = sin(ZETA*X)*sin(ZETA*Y) + 0.05*(X^2 + Y^2)
# ZETA=20, all variables free, start=(-506,-506.2)
#
# HMP element: F(X,Y,A) = sin(A*X)*sin(A*Y)  (A=ZETA)
# SQ element:  F(X) = X^2
# Group OBJ (N type, identity): H*1 + SX*0.05 + SY*0.05
function humps_model(; T = Float64, backend = nothing, kwargs...)
    zeta = T(20)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2;
        lvar  = T[-Inf, -Inf],
        uvar  = T[Inf, Inf],
        start = T[-506.0, -506.2])
    ExaModels.@add_obj(c,
        sin(zeta * x[1]) * sin(zeta * x[2]) + T(0.05) * x[1]^2 + T(0.05) * x[2]^2)
    return ExaModels.ExaModel(c; kwargs...)
end

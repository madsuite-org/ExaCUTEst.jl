# JANNSON4: Convex-concave extension example 4 (Jannson 2000)
# N=10000 variables x[1]..x[N] with bounds [-1,1]
# SIF GROUPS:
#   XN G X2 -1.0   → G linear = -x[2]
#   XN G0 X1 1.0   → G0 linear = +x[1]
#   DO I 1 N; XN G(I) X(I) 1.0  → G(I) linear = x[I]
#   XL Q, XL P
# CONSTANTS: G0=1.0, G(I)=1.0, Q=1.0, P=0.1
# GROUP TYPE: G→L22, G(I)→L2, Q→identity (XL), P→identity (XL)
# GROUP USES:
#   XT G L22; XE G E(1)=x[1]^2
#   DO I 0 N; XT G(I) L2
#   DO I 1 N; XE Q E(I)=x[I]^2
#   XE P P = x[1]*x[2]
#
# Element P has IV: XX=X(1)=x[1], YY=X(2)=x[2], F=x[1]*x[2]
# Wait, SIF: XT P PROD, ZV P X X1, ZV P Y X2 → F=X*Y=x[1]*x[2]
#
# Objective:
#   G (L22): 0.5*(x[1]^2 - x[2])^2
#   G(0) (L2): (x[1] - 1)^2
#   G(I) for I=1..N (L2): (x[I] - 1)^2
# Inequality constraints (L type, value >= 0):
#   Q: sum_{I=1..N} x[I]^2 - 1.0 >= 0
#   P: x[1]*x[2] - 0.1 >= 0
function jannson4_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-1), uvar = T(1))

    # Group G (L22): 0.5*(x[1]^2 - x[2])^2
    ExaModels.@add_obj(c, T(0.5) * (x[1]^2 - x[2])^2)

    # Group G0 (L2): (x[1] - 1)^2
    ExaModels.@add_obj(c, (x[1] - T(1))^2)

    # Group G(I) for I=1..N (L2): (x[I] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N)

    # Inequality constraint Q: sum x[I]^2 >= 1.0
    ExaModels.@add_con(c, sum(x[i]^2 for i in 1:N) - T(1); lcon = T(0), ucon = T(Inf))

    # Inequality constraint P: x[1]*x[2] >= 0.1
    ExaModels.@add_con(c, x[1] * x[2] - T(0.1); lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

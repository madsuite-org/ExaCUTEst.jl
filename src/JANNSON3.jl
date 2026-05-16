# JANNSON3: Convex-concave extension example 3 (Jannson 2000)
# N=10000, 2N variables x[1]..x[2N] with bounds [-1,1]
# Objective composed of groups:
#   G (L22 type): group value = x[2] - x[2]*1 - 1.0 = x[2] - 1... wait
#   Group G: linear part = -x[2]*1.0 = -X(2) coeff -1 but also element E(1)=x[1]^2
#   Actually from SIF: XN G X2 -1.0, then XT G L22, XE G E(1) where E(1)=X(1)^2
#   L22 group type: F(v) = 0.5*v^2, so L22 contributes 0.5*v^2 (not v^2)
#   group G value = v_G = -x[2] + x[1]^2 - C(G) = -x[2] + x[1]^2 - 0 (no CONSTANT for G, X2 coeff -1 means linear -x[2])
#   Wait: XN G X2 -1.0 means variable X(2) with coeff -1.0 in group G (linear part)
#   Actually in SIF notation variable index starts from 1 so X(2) is x[2]
#   Group G linear: -x[2]. Element E(1) = x[1]^2 added with default coeff 1.0
#   So group G value = -x[2] + x[1]^2
#   L22 contribution = 0.5*(-x[2] + x[1]^2)^2
#
#   G(0): L2 type, linear -1.0 + x[1]*1.0, no elements (from XN G(0) X1 1.0, const=1)
#   Actually from SIF: XN G0 X1 1.0, CONSTANT = G0 1.0, XT G(I) L2
#   G(I) for I=0..2N: XT G(I) L2, XN G(I) X(I) 1.0, constant = 1.0
#   → contribution = (x[I] - 1.0)^2
#   But G(0) has X1 1.0... wait SIF uses: DO I 1 2N; XN G(I) X(I) 1.0; XT G(I) L2
#   so G(I) for I=1..2N: value = x[I] - 1, contribution = (x[I]-1)^2
#   G(0): XN G0 X1 1.0 (not in DO loop), const = 1.0 → value = x[1]-1, contribution = (x[1]-1)^2
#   But G(0) is separate, and DO I 0 2N XT G(I) L2 includes I=0
#
#   Q (L type, lcon=0): sum_{I=1}^{2N} E(I) = sum x[I]^2, const = 0.75
#   → sum x[I]^2 - 0.75 >= 0
#
#   P (L type, lcon=0): sum_{I=1}^N P(I) = sum x[I]*x[N+I], const = 1/5N
#   → sum_{I=1}^N x[I]*x[N+I] - 1/(5N) >= 0
#
# Careful analysis:
# SIF GROUPS:
#   XN G X2 -1.0   → G has linear term -X(2) = -x[2]
#   XN G0 X1 1.0   → G0 has linear term +X(1) = +x[1]
#   DO I 1 2N; XN G(I) X(I) 1.0; XE L X(I) 1.0  → G(I) linear = x[I], L linear += x[I]
#   XL Q           → Q is L constraint (≥0)
#   XL P           → P is L constraint (≥0)
# CONSTANTS:
#   G0 = 1.0, G(I) = 1.0 for I=1..2N, L = 1.0, Q = 0.75, P = 1/5N
# GROUP USES:
#   XT G L22; XE G E(1)  → G has L22 type and element E(1) = X(1)^2
#   DO I 0 2N; XT G(I) L2  → all G(I) have L2 type
#   DO I 1 2N; XE Q E(I)   → Q += x[I]^2
#   DO I 1 N; XE P P(I)   → P += x[I]*x[N+I]
#
# So objective:
#   G (L22, no scale): 0.5*(-x[2] + x[1]^2)^2
#   G(0) (L2, no scale): (x[1] - 1.0)^2
#   G(I) for I=1..2N (L2, no scale): (x[I] - 1.0)^2
# Inequality constraints (L type, value >= 0):
#   Q: sum_{I=1..2N} x[I]^2 - 0.75 >= 0
#   P: sum_{I=1..N} x[I]*x[N+I] - 1/(5N) >= 0
# Equality constraint:
#   L: sum_{I=1..2N} x[I] - 1.0 = 0
function jannson3_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    nvar = 2 * N
    inv5N = T(1) / T(5 * N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = T(-1), uvar = T(1))

    # Group G (L22 type, no scale): 0.5*(−x[2] + x[1]²)²
    # v_G = x[1]^2 - x[2], L22: F(v) = 0.5*v^2
    ExaModels.@add_obj(c, T(0.5) * (x[1]^2 - x[2])^2)

    # Group G0 (L2 type, no scale): (x[1] - 1)^2
    ExaModels.@add_obj(c, (x[1] - T(1))^2)

    # Group G(I) for I=1..2N (L2 type): (x[I] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:nvar)

    # Equality constraint L: sum_{I=1..2N} x[I] = 1
    ExaModels.@add_con(c, sum(x[i] for i in 1:nvar) - T(1))

    # Inequality constraint Q: sum x[I]^2 >= 0.75
    ExaModels.@add_con(c, sum(x[i]^2 for i in 1:nvar) - T(0.75); lcon = T(0), ucon = T(Inf))

    # Inequality constraint P: sum_{I=1..N} x[I]*x[N+I] >= 1/(5N)
    ExaModels.@add_con(c, sum(x[i] * x[N+i] for i in 1:N) - inv5N; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

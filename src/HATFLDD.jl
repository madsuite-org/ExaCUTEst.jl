# HATFLDD: Hatfield polytechnic exponential fitting test problem
# 3 free variables X1, X2, X3; start=[1,-1,0]
# Minimize sum_{I=1..10} (exp(T(I)*X3) - X1*exp(T(I)*X2) + Z(I))^2
#
# ELEMENT TYPES:
#   XPEXP(X=X1, Y=X2; T) = X1 * exp(T*X2)     [coefficient -1 in group]
#   PEXP(X=X3; T)         = exp(T*X3)           [coefficient +1 in group]
#
# GROUP G(I): v = -XPEXP + PEXP - rhs
#   CONSTANTS: rhs = -Z(I)  (via AM -ZI = -1.0*Z(I); Z G(I) -ZI → rhs = -Z(I))
#   v = -X1*exp(T*X2) + exp(T*X3) - (-Z(I)) = exp(T*X3) - X1*exp(T*X2) + Z(I)
#   GROUP TYPE L2: contribution = v^2
#
# All variables free (FR DEFAULT); no constraints.
function hatfldd_model(; T = Float64, backend = nothing, kwargs...)
    τ = T[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    z = T[1.751, 1.561, 1.391, 1.239, 1.103, 0.981, 0.925, 0.8721, 0.8221, 0.7748]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar  = T[-Inf, -Inf, -Inf],
                       uvar  = T[Inf, Inf, Inf],
                       start = T[1.0, -1.0, 0.0])

    ExaModels.@add_par(c, τp, τ)
    ExaModels.@add_par(c, zp, z)

    # G(I): L2 group, v = exp(T(I)*X3) - X1*exp(T(I)*X2) + Z(I)
    ExaModels.@add_obj(c,
        (exp(τp[i] * x[3]) - x[1] * exp(τp[i] * x[2]) + zp[i])^2
        for i in 1:10)

    return ExaModels.ExaModel(c; kwargs...)
end

# HATFLDE: Hatfield polytechnic exponential fitting test problem (NG=21)
# 3 free variables X1, X2, X3; start=[1,-1,0]
# Minimize sum_{I=1..21} (exp(T(I)*X3) - X1*exp(T(I)*X2) + Z(I))^2
#
# Same structure as HATFLDD but with 21 data points (T and Z values differ).
#
# ELEMENT TYPES:
#   XPEXP(X=X1, Y=X2; T) = X1 * exp(T*X2)     [coefficient -1 in group]
#   PEXP(X=X3; T)         = exp(T*X3)           [coefficient +1 in group]
#
# GROUP G(I): v = exp(T(I)*X3) - X1*exp(T(I)*X2) + Z(I)
# GROUP TYPE L2: contribution = v^2
#
# All variables free (FR DEFAULT); no constraints.
function hatflde_model(; T = Float64, backend = nothing, kwargs...)
    τ = T[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
          0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    z = T[1.561, 1.473, 1.391, 1.313, 1.239, 1.169, 1.103, 1.04, 0.981, 0.925,
          0.8721, 0.8221, 0.7748, 0.73, 0.6877, 0.6477, 0.6099, 0.5741, 0.5403, 0.5084, 0.4782]

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
        for i in 1:21)

    return ExaModels.ExaModel(c; kwargs...)
end

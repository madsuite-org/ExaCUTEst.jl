# HATFLDF: Hatfield polytechnic test problem (3 variables, 3 equality constraints)
# 3 free variables X1, X2, X3; start = 0.1
# 3 equality constraint groups (E type, no group type → identity):
#   G(I): v = X1 + XPEXP(X2, X3; T=I) - RHS[I] = 0
#   Element XPEXP(X=X2, Y=X3; T=I) = X2 * exp(T*X3) = X2 * exp(I*X3)
#   RHS: G1=0.032, G2=0.056, G3=0.099
#   constraint: X1 + X2*exp(I*X3) - RHS[I] = 0
function hatfldf_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(0.1))
    ExaModels.@add_par(c, RHS, T[0.032, 0.056, 0.099])
    ExaModels.@add_con(c, x[1] + x[2] * exp(T(i) * x[3]) - RHS[i] for i in 1:3)
    return ExaModels.ExaModel(c; kwargs...)
end

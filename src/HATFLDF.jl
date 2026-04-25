# HATFLDF: Hatfield polytechnic test problem (3 variables, 3 equality constraints)
# CUTEst variable ordering (empirically verified): x[1]=X3, x[2]=X2, x[3]=X1
# 3 free variables; all start = 0.1
# 3 equality constraint groups (E type, no group type → identity):
#   G(I): v = X1 + XPEXP(X2, X3; T=I) - RHS[I] = 0
#   Element XPEXP(X=X2, Y=X3; T=I) = X2 * exp(T*X3) = X2 * exp(I*X3)
#   RHS: G1=0.032, G2=0.056, G3=0.099
#   constraint (in ExaModel vars): x[3] + x[2]*exp(RI[i]*x[1]) - RHS[i] = 0
function hatfldf_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(0.1))
    ExaModels.@add_par(c, RHS, T[0.032, 0.056, 0.099])
    ExaModels.@add_par(c, RI, T[1.0, 2.0, 3.0])
    ExaModels.@add_con(c, x[3] + x[2] * exp(RI[i] * x[1]) - RHS[i] for i in 1:3)
    return ExaModels.ExaModel(c; kwargs...)
end

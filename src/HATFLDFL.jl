# HATFLDFL: Fletcher's variation of HATFLDF (3 variables, unconstrained, L2)
# 3 free variables X1, X2, X3; start = [1.2, -1.2, 0.98]
# 3 objective groups (N type, L2 group type):
#   G(I): element XPEXP(X=X2, Y=X3; T=I) = X2 * X3^I  (power, not exp: Y**IT in SIF)
#   v = X1 + X2*X3^I - RHS[I]
#   contribution = v^2 (L2)
#   RHS: G1=0.032, G2=0.056, G3=0.099
function hatfldfl_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[1.2, -1.2, 0.98])
    ExaModels.@add_par(c, RHS, T[0.032, 0.056, 0.099])
    ExaModels.@add_par(c, RI, T[1.0, 2.0, 3.0])
    ExaModels.@add_obj(c, (x[1] + x[2] * x[3]^RI[i] - RHS[i])^2 for i in 1:3)
    return ExaModels.ExaModel(c; kwargs...)
end

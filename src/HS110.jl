# HS110: Logarithmic and exponential sum objective (10 variables, unconstrained)
# Source: Hock & Schittkowski, problem 110
# SIF: 2n+1 groups, LOGSQR group type for 2LOG/10LOG, EXP group type for GEXP
#   2LOG(I): LOGSQR, linear X(I), constant=2  → log(x[i]-2)^2
#   10LOG(I): LOGSQR, linear -X(I), constant=-10 → log(10-x[i])^2
#   GEXP: EXP type F(v)=-exp(0.2*v), elements LOGX(I)=log(X(I)) → -exp(0.2*sum log(x[i]))
function hs110_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 10;
        lvar  = T[2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001],
        uvar  = T[9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999],
        start = T[9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0  ])
    # LOGSQR groups: sum log(x[i]-2)^2 + log(10-x[i])^2
    ExaModels.@add_obj(c, log(x[i] - T(2))^2 + log(T(10) - x[i])^2 for i in 1:10)
    # GEXP group: F(v) = -exp(0.2*v), v = sum_i log(x[i])
    # = -exp(0.2 * log(prod x[i])) = -prod x[i]^0.2
    ExaModels.@add_obj(c, -(x[1]^T(0.2) * x[2]^T(0.2) * x[3]^T(0.2) * x[4]^T(0.2) * x[5]^T(0.2) *
                             x[6]^T(0.2) * x[7]^T(0.2) * x[8]^T(0.2) * x[9]^T(0.2) * x[10]^T(0.2)))
    return ExaModels.ExaModel(c; kwargs...)
end

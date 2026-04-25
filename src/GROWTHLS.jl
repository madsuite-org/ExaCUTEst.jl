# GROWTHLS: Growth model fitting, f = sum (U1*n^(U2+log(n)*U3) - Y_n)^2
# 3 free vars (U1=100, U2=U3=0), 12 observation groups (L2 type)
# FIT element: U1 * RN^(U2 + log(RN)*U3) = U1 * exp(log(n)*(U2 + log(n)*U3))
function growthls_model(; T = Float64, backend = nothing, kwargs...)
    ns = T[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 25.0]
    ys = T[8.0, 8.4305, 9.5294, 10.4627, 12.0, 13.0205, 14.5949, 16.1078, 18.0596, 20.4569, 24.25, 32.9863]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T[T(100), T(0), T(0)])
    ExaModels.@add_par(c, LOGNI, log.(ns))
    ExaModels.@add_par(c, YI, ys)
    ExaModels.@add_obj(c,
        (x[1] * exp(LOGNI[i] * (x[2] + LOGNI[i] * x[3])) - YI[i])^2 for i in 1:12)
    return ExaModels.ExaModel(c; kwargs...)
end

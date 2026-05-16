# QPNBAND: Same as QPBAND (banded QP, same structure)
function qpnband_model(N = 50000; T = Float64, backend = nothing, kwargs...)
    M = N ÷ 2
    RN = T(N)

    lvar = zeros(T, N)
    uvar = fill(T(2.0), N)
    start = zeros(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_obj(c, (-T(i) / RN) * x[i] for i in 1:N)
    ExaModels.@add_obj(c, x[i]^2 for i in 1:N)
    ExaModels.@add_obj(c, T(-0.5) * x[i] * x[i+1] for i in 1:N-1)

    ExaModels.@add_con(c, x[i] + x[M+i] - T(1.0) for i in 1:M;
                       lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

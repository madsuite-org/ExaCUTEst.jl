function _diag_qe_model(N, shift; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N;
        lvar = fill(T(-100000), N),
        uvar = fill(T(1000000), N),
        start = ones(T, N))
    data = [(i = Int32(i), h = T(i + shift)) for i in 1:N]
    ExaModels.@add_obj(c, x[d.i] + T(0.5) * d.h * x[d.i]^2 for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end

function _diag_qb_model(N, shift, sign; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N;
        lvar = fill(T(-100000), N),
        uvar = fill(T(1000000), N),
        start = ones(T, N))
    rn = Float64(N)
    data = [(i = Int32(i), h = T(sign * Float64(i)^2 / rn + shift)) for i in 1:N]
    ExaModels.@add_obj(c, x[d.i] + T(0.5) * d.h * x[d.i]^2 for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end

diagpqe_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qe_model(N, 0.0; T, backend, kwargs...)

diagnqe_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qe_model(N, -Float64(N) - 1.0; T, backend, kwargs...)

diagiqe_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qe_model(N, -Float64(N) / 2.0; T, backend, kwargs...)

diagpqb_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, 0.0, 1.0; T, backend, kwargs...)

diagnqb_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, -(Float64(N) + 1.0/Float64(N)), 1.0; T, backend, kwargs...)

diagpqt_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, Float64(N) + 1.0/Float64(N), -1.0; T, backend, kwargs...)

diagiqb_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, 1.0/Float64(N) - Float64(N)/2.0, 1.0; T, backend, kwargs...)

diagiqt_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, 1.0/Float64(N) + Float64(N)/2.0, -1.0; T, backend, kwargs...)

diagnqt_model(N = 1000; T = Float64, backend = nothing, kwargs...) =
    _diag_qb_model(N, 0.0, -1.0; T, backend, kwargs...)

# READING3: Nonlinear optimal control with periodic boundary condition, N=2000 default
# Source: Lyle & Nichols (1991)
# Same as READING1 but:
#   - X(0..N) in [-0.5, 0.5] (no fixed X(0))
#   - Extra constraint: X(0) - X(N) = 0 (periodicity)
# Julia: x[2i+1]=X(i), x[2i+2]=U(i), for i=0..N
function reading3_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    PI  = T(3.141592653589)
    A   = T(0.07716)
    H   = T(1) / T(N)
    HALF_A = T(1) / (T(2) * A)

    c2pit = T[cos(T(2) * PI * T(i) * H) for i in 0:N]

    lvar = Vector{T}(undef, 2 * (N + 1))
    uvar = Vector{T}(undef, 2 * (N + 1))
    start = zeros(T, 2 * (N + 1))
    for i in 0:N
        lvar[2i + 1] = T(-0.5); uvar[2i + 1] = T(0.5)   # X(i)
        lvar[2i + 2] = T(0.0);  uvar[2i + 2] = T(1.0)   # U(i)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2 * (N + 1); lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, ct, c2pit)

    # Objective: same as READING1
    ExaModels.@add_obj(c,
        -(H / T(2)) * x[2d.I + 2] * (x[2d.I + 1] - ct[d.I + 1])^2 -
        (H / T(2)) * x[2d.I] * (x[2d.I - 1] - ct[d.I])^2
        for d in [(I = I,) for I in 1:N])

    # Constraints C(I) for I=1..N (same as READING1)
    ExaModels.@add_con(c,
        x[2d.I + 1] / H - x[2d.I - 1] / H +
        HALF_A * x[2d.I + 2] * (x[2d.I + 1] - ct[d.I + 1]) +
        HALF_A * x[2d.I] * (x[2d.I - 1] - ct[d.I])
        for d in [(I = I,) for I in 1:N])

    # Periodicity constraint: X(0) - X(N) = 0
    # X(0) = x[1], X(N) = x[2N+1]
    ExaModels.@add_con(c, x[1] - x[2N + 1])

    return ExaModels.ExaModel(c; kwargs...)
end

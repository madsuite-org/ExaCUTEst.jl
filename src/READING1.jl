# READING1: Nonlinear optimal control (tide modelling), N=2000 default
# Source: Lyle & Nichols (1991)
# Variables: X(0)..X(N), U(0)..U(N) (interleaved in SIF order)
# Julia: x[2i+1] = X(i), x[2i+2] = U(i), for i=0..N
#
# Objective (N type, identity): sum_{I=1}^N I(I) where
#   I(I) = -(H/2)*U(I)*(X(I)-cos(2π*I*H))^2 - (H/2)*U(I-1)*(X(I-1)-cos(2π*(I-1)*H))^2
#
# Constraints C(I) for I=1..N (E type, equality):
#   X(I)/H - X(I-1)/H + (1/2A)*U(I)*(X(I)-cos(2π*I*H)) + (1/2A)*U(I-1)*(X(I-1)-cos(2π*(I-1)*H)) = 0
#
# Bounds: X(0) fixed at 0.25; X(1..N) in [-0.5, 0.5]; U(0..N) in [0, 1]
function reading1_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    PI  = T(3.141592653589)
    A   = T(0.07716)
    H   = T(1) / T(N)
    HALF_A = T(1) / (T(2) * A)

    # cos values: c2pit[i] = cos(2*pi*i*H) for i=0..N
    c2pit = T[cos(T(2) * PI * T(i) * H) for i in 0:N]

    # Variable ordering: x[2i+1] = X(i), x[2i+2] = U(i), i=0..N
    lvar = Vector{T}(undef, 2 * (N + 1))
    uvar = Vector{T}(undef, 2 * (N + 1))
    start = Vector{T}(undef, 2 * (N + 1))
    for i in 0:N
        # X(i)
        if i == 0
            lvar[2i + 1] = T(0.25); uvar[2i + 1] = T(0.25)
        else
            lvar[2i + 1] = T(-0.5); uvar[2i + 1] = T(0.5)
        end
        start[2i + 1] = T(0)
        # U(i)
        lvar[2i + 2] = T(0.0); uvar[2i + 2] = T(1.0)
        start[2i + 2] = T(0)
    end
    start[1] = T(0.25)  # X(0) fixed

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2 * (N + 1); lvar = lvar, uvar = uvar, start = start)

    # Precompute cos values as parameters
    ExaModels.@add_par(c, ct, c2pit)

    # Objective: sum_{I=1}^N I(I)
    # I(I) = -(H/2)*U(I)*(X(I)-ct[I+1])^2 - (H/2)*U(I-1)*(X(I-1)-ct[I])^2
    # x[2I+1] = X(I), x[2I+2] = U(I), x[2I-1] = X(I-1), x[2I] = U(I-1)
    ExaModels.@add_obj(c,
        -(H / T(2)) * x[2d.I + 2] * (x[2d.I + 1] - ct[d.I + 1])^2 -
        (H / T(2)) * x[2d.I] * (x[2d.I - 1] - ct[d.I])^2
        for d in [(I = I,) for I in 1:N])

    # Constraints C(I) for I=1..N
    ExaModels.@add_con(c,
        x[2d.I + 1] / H - x[2d.I - 1] / H +
        HALF_A * x[2d.I + 2] * (x[2d.I + 1] - ct[d.I + 1]) +
        HALF_A * x[2d.I] * (x[2d.I - 1] - ct[d.I])
        for d in [(I = I,) for I in 1:N])

    return ExaModels.ExaModel(c; kwargs...)
end

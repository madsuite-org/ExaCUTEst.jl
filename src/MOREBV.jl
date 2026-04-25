# MOREBV: Boundary Value problem (N=5000 variables, unconstrained)
# Source: More', Garbow, Hillstrom (1981), problem 28; Buckley#17
# SIF: N L2 groups (no scale), element WCUBE: (V+B)^3, coefficient HALFH2 = 0.5*h^2
# h = 1/(N+1), B_I = I*h + 1
#
# Group values:
#   v_G(1)   = 2*x[1] - x[2]      + halfh2*(x[1] + h + 1)^3
#   v_G(I)   = -x[I-1]+2*x[I]-x[I+1] + halfh2*(x[I] + I*h+1)^3  for I=2..N-1
#   v_G(N)   = -x[N-1] + 2*x[N]   + halfh2*(x[N] + N*h+1)^3
# Objective = sum_{I=1}^{N} v_G(I)^2
#
# Variables: all free (FR DEFAULT). Start: x_I = I*h*(I*h - 1).
function morebv_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    h = T(1) / T(N + 1)
    halfh2 = T(0.5) * h^2
    B = T[i * h + T(1) for i in 1:N]
    start = T[i * h * (i * h - T(1)) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, B, B)
    # I=1: v = 2*x[1] - x[2] + halfh2*(x[1]+B[1])^3
    ExaModels.@add_obj(c, (T(2) * x[1] - x[2] + halfh2 * (x[1] + B[1])^3)^2)
    # I=2..N-1: v = -x[I-1] + 2*x[I] - x[I+1] + halfh2*(x[I]+B[I])^3
    ExaModels.@add_obj(
        c,
        (-x[i-1] + T(2) * x[i] - x[i+1] + halfh2 * (x[i] + B[i])^3)^2
        for i in 2:N-1
    )
    # I=N: v = -x[N-1] + 2*x[N] + halfh2*(x[N]+B[N])^3
    ExaModels.@add_obj(c, (-x[N-1] + T(2) * x[N] + halfh2 * (x[N] + B[N])^3)^2)
    return ExaModels.ExaModel(c; kwargs...)
end

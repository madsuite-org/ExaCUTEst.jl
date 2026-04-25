# FLETCBV2: Boundary value problem from Fletcher (1992), N=5000 variables, unconstrained
# Discretization of x'' = -2 + sin(x) on [0,1] with x(0)=0, x(1)=1
# multiplied by h^2 before integrating.
#
# SIF structure (H=1/(N+1), H2=H^2, KAPPAH2=H2):
#   G(0):   HALFL2, v=x[1]         → 0.5*x[1]^2
#   G(I):   HALFL2, v=x[I]-x[I+1] → 0.5*(x[I]-x[I+1])^2  for I=1..N-1
#   G(N):   HALFL2, v=x[N]         → 0.5*x[N]^2
#   L(I):   identity, v=-2H2*x[I]  → -2H2*x[I]             for I=1..N-1
#   L(N):   identity, v=(-1-2H2)*x[N] → (-1-2H2)*x[N]
#   C(I):   identity, element -H2*cos(x[I]) → -H2*cos(x[I]) for I=1..N
#   Bounds: FR DEFAULT → all variables free
#   Start: X(I) = I*H
function fletcbv2_model(; T = Float64, backend = nothing, kwargs...)
    N = 5000
    H = T(1) / T(N + 1)
    H2 = H * H
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                       start = T[i * H for i in 1:N])
    # G(0): 0.5*x[1]^2
    ExaModels.@add_obj(c, T(0.5) * x[1]^2)
    # G(I) for I=1..N-1: 0.5*(x[I]-x[I+1])^2
    ExaModels.@add_obj(c, T(0.5) * (x[i] - x[i+1])^2 for i in 1:N-1)
    # G(N): 0.5*x[N]^2
    ExaModels.@add_obj(c, T(0.5) * x[N]^2)
    # L(I) for I=1..N-1: -2H2*x[I]
    ExaModels.@add_obj(c, -T(2) * H2 * x[i] for i in 1:N-1)
    # L(N): (-1-2H2)*x[N]
    ExaModels.@add_obj(c, -(T(1) + T(2) * H2) * x[N])
    # C(I) for I=1..N: -H2*cos(x[I])
    ExaModels.@add_obj(c, -H2 * cos(x[i]) for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end

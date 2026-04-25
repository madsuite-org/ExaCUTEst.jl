# FLETCHBV: Boundary value problem from Fletcher (1992), N=5000 variables, unconstrained
# "Incorrect" formulation (see FLETCBV2.SIF for the correct version).
#
# SIF structure (H=1/(N+1), H2=H^2, 1/H2=(N+1)^2, KAPPA=1.0, OBJSCALE=1.0):
#   G(0):   HALFL2 (F=0.5*v^2), v=x[1]           → 0.5*x[1]^2
#   G(I):   HALFL2, v=x[I]-x[I+1]                 → 0.5*(x[I]-x[I+1])^2  for I=1..N-1
#   G(N):   HALFL2, v=x[N]                         → 0.5*x[N]^2
#   L(I):   identity, v=-2/H2*x[I]                → -2*(N+1)^2*x[I]       for I=1..N-1
#   L(N):   identity, v=+2/H2*x[N]                → +2*(N+1)^2*x[N]
#     (SIF: RM -1-2/H2 -2/H2 -1.0 → -2/H2 * (-1.0) = +2/H2, label is misleading)
#   C(I):   identity, element -KAPPA/H2*cos(x[I]) → -(N+1)^2*cos(x[I])   for I=1..N
#   Bounds: FR DEFAULT → all variables free
#   Start: x[I] = I*H = I/(N+1)
function fletchbv_model(; T = Float64, backend = nothing, kwargs...)
    N = 5000
    H = T(1) / T(N + 1)
    inv_H2 = T(N + 1)^2   # 1/H^2 = (N+1)^2
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf),
                       start = T[i * H for i in 1:N])
    # G(0): 0.5*x[1]^2
    ExaModels.@add_obj(c, T(0.5) * x[1]^2)
    # G(I) for I=1..N-1: 0.5*(x[I]-x[I+1])^2
    ExaModels.@add_obj(c, T(0.5) * (x[i] - x[i+1])^2 for i in 1:N-1)
    # G(N): 0.5*x[N]^2
    ExaModels.@add_obj(c, T(0.5) * x[N]^2)
    # L(I) for I=1..N-1: -2/H2*x[I]
    ExaModels.@add_obj(c, -T(2) * inv_H2 * x[i] for i in 1:N-1)
    # L(N): +2/H2*x[N]  (RM gives -2/H2 * (-1.0) = +2/H2)
    ExaModels.@add_obj(c, T(2) * inv_H2 * x[N])
    # C(I) for I=1..N: -KAPPA/H2*cos(x[I]) = -(N+1)^2*cos(x[I])
    ExaModels.@add_obj(c, -inv_H2 * cos(x[i]) for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end

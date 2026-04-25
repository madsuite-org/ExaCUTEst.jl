# OSCIGRNE: roots of gradient of Nesterov's oscillating path problem
# Nonlinear equations version, N=100000 variables, N equality constraints
# Source: Nick Gould, June 2011
# SIF classification: NOR2-AN-V-V
#
# RHO=500, 2RHO=1000, -4RHO=-2000
# Element A(V,U,P): F = P*(V - 2*U^2 + 1)  [V=X(I), U=X(I-1), P=2RHO=1000]
# Element B(V,U,P): F = P*(V - 2*U^2 + 1)*U [V=X(I+1), U=X(I), P=-4RHO=-2000]
#
# Constraints (all E type, lcon=ucon=0):
# G1:       0.5*x[1] - 0.5 + B(V=x[2], U=x[1], P=-4000) = 0
# G(I) I=2..N-1: A(I,V=x[I],U=x[I-1],P=2000) + B(I,V=x[I+1],U=x[I],P=-4000) = 0
# G(N):     A(N,V=x[N],U=x[N-1],P=2000) = 0
function oscigrne_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    p2rho = T(1000)
    pm4rho = T(-2000)
    start = [T(-2); fill(T(1), N - 1)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # G1: 0.5*x[1] - 0.5 + B(V=x[2], U=x[1], P=-4000) = 0
    # B = pm4rho*(x[2] - 2*x[1]^2 + 1)*x[1]
    ExaModels.@add_con(c, T(0.5) * x[1] - T(0.5) + pm4rho * (x[2] - T(2) * x[1]^2 + T(1)) * x[1])
    # G(I) for I=2..N-1: A(I) + B(I) = 0
    # A(I) = 2000*(x[I] - 2*x[I-1]^2 + 1)
    # B(I) = -4000*(x[I+1] - 2*x[I]^2 + 1)*x[I]
    ExaModels.@add_con(
        c,
        p2rho * (x[i] - T(2) * x[i-1]^2 + T(1)) + pm4rho * (x[i+1] - T(2) * x[i]^2 + T(1)) * x[i]
        for i in 2:N-1
    )
    # G(N): A(N) = 0
    # A(N) = 2000*(x[N] - 2*x[N-1]^2 + 1)
    ExaModels.@add_con(c, p2rho * (x[N] - T(2) * x[N-1]^2 + T(1)))
    return ExaModels.ExaModel(c; kwargs...)
end

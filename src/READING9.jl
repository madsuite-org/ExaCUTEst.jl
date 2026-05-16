# READING9: Nonlinear optimal control, N=5000 default
# Source: Gould & Toint (1995), variant of Lyle & Nichols problem
# Variables: P(0..N), U(0..N) — state and control
# Julia: x[i+1] = P(i) for i=0..N; x[N+2+i] = U(i) for i=0..N
#
# Objective OBJ (N type):
#   OBJ = sum_{I=0}^{N-1} [ -H/2 * (U(I)*P(I))^2 - H/2 * (U(I+1)*P(I+1))^2 ]
#       = -H/2 * [ (U(0)*P(0))^2 + 2*(U(1)*P(1))^2 + ... + 2*(U(N-1)*P(N-1))^2 + (U(N)*P(N))^2 ]
#
# Constraints S(I) for I=0..N-1 (E type):
#   P(I+1) - (1 + K1H)*P(I) + K2H*U(I)*P(I) = H*sin(I*H)
#
# Bounds: P(0) fixed at 0.0; P(1..N) free; U(0..N) in [0, 1]
function reading9_model(Np1 = 5001; T = Float64, backend = nothing, kwargs...)
    N  = Np1 - 1
    K1 = T(0.07716)
    K2 = T(0.07716)
    H  = T(1) / T(N)
    K1H   = K1 * H
    K1H1  = T(1) + K1H   # K1H+1
    K2H   = K2 * H
    HH2   = H / T(2)      # H/2

    # Precompute RHS: H*sin(I*H) for I=0..N-1
    rhs = T[H * sin(T(i) * H) for i in 0:N-1]

    # Variables: P(0..N), then U(0..N)
    # x[i+1] = P(i), x[N+2+i] = U(i)
    nv = 2 * (N + 1)
    lvar = Vector{T}(undef, nv)
    uvar = Vector{T}(undef, nv)
    start = fill(T(0.2), nv)
    # P(0..N): P(0) fixed at 0.0, rest free
    for i in 0:N
        lvar[i + 1] = T(-Inf)
        uvar[i + 1] = T(Inf)
    end
    lvar[1] = T(0.0); uvar[1] = T(0.0); start[1] = T(0.0)
    # U(0..N): [0, 1]
    for i in 0:N
        lvar[N + 2 + i] = T(0.0)
        uvar[N + 2 + i] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, rh, rhs)

    # P(i) = x[i+1], U(i) = x[N+2+i]
    # Objective: -H/2 * (U(0)*P(0))^2 (endpoint, appears once)
    ExaModels.@add_obj(c, -HH2 * (x[N + 2] * x[1])^2)
    # Middle terms (appear twice): -H * (U(i)*P(i))^2 for i=1..N-1
    ExaModels.@add_obj(c,
        -H * (x[N + 2 + d.i] * x[d.i + 1])^2
        for d in [(i = i,) for i in 1:N-1])
    # Endpoint N: -H/2 * (U(N)*P(N))^2
    ExaModels.@add_obj(c, -HH2 * (x[N + 2 + N] * x[N + 1])^2)

    # Constraints S(I) for I=0..N-1:
    # P(I+1) - K1H1*P(I) + K2H*U(I)*P(I) - H*sin(I*H) = 0
    # x[I+2] - K1H1*x[I+1] + K2H*x[N+2+I]*x[I+1] - rh[I+1] = 0
    ExaModels.@add_con(c,
        x[d.I + 2] - K1H1 * x[d.I + 1] + K2H * x[N + 2 + d.I] * x[d.I + 1] - rh[d.I + 1]
        for d in [(I = I,) for I in 0:N-1])

    return ExaModels.ExaModel(c; kwargs...)
end

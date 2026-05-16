# READING4: Nonlinear optimal control, variant of READING1, N=5000 default
# Source: Toint (1992), variant of Lyle & Nichols problem
# N+1 variables X(0)..X(N)
# Objective J (N type, SCALE=1/A, identity for N groups so scale ignored):
#   J = -1*ENERGY(1) - 2*ENERGY(2) - ... - 2*ENERGY(N-1) - 1*ENERGY(N)
#   ENERGY(I): F = (cos(2π*I*H) - X(I)) * (X(I) - X(I-1))
# Constraints U(I) for I=1..N (G type, >= 0):
#   (2A/H) * UC(I) >= 0  (actually 0 <= UC(I) <= 1 from RANGES)
#   UC(I): C = (X(I) - X(I-1)) / (cos(2π*I*H) - X(I))  -> value is C
#   RANGES: lcon=0, ucon=1
# Bounds: X(0) fixed at 0.25; X(1..N) in [-0.5, 0.5]
function reading4_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    PI   = T(3.141592653589)
    A    = T(0.07716)
    H    = T(1) / T(N)
    HA   = H / (T(2) * A)   # H/2A = H/(2A)
    TWO_A_H = T(2) * A / H  # 2A/H

    c2pit = T[cos(T(2) * PI * T(i) * H) for i in 1:N]  # cos(2π*I*H) for I=1..N

    # Variables X(0)..X(N): Julia x[i+1] = X(i)
    lvar = Vector{T}(undef, N + 1)
    uvar = Vector{T}(undef, N + 1)
    start = zeros(T, N + 1)
    lvar[1] = T(0.25); uvar[1] = T(0.25); start[1] = T(0.25)
    for i in 1:N
        lvar[i + 1] = T(-0.5); uvar[i + 1] = T(0.5)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, ct, c2pit)

    # Objective: J = ENERGY(1)*(-1) + sum_{I=2}^{N-1} ENERGY(I)*(-2) + ENERGY(N)*(-1)
    # ENERGY(I) = (ct[I] - X(I)) * (X(I) - X(I-1))
    # = (ct[I] - x[I+1]) * (x[I+1] - x[I])
    # Scale=1/A for N group but per CLAUDE.md, scale is ignored for N identity groups.
    ExaModels.@add_obj(c, -(ct[1] - x[2]) * (x[2] - x[1]))
    ExaModels.@add_obj(c, T(-2) * (ct[d.I] - x[d.I + 1]) * (x[d.I + 1] - x[d.I])
                       for d in [(I = I,) for I in 2:N-1])
    ExaModels.@add_obj(c, -(ct[N] - x[N + 1]) * (x[N + 1] - x[N]))

    # Constraints U(I) for I=1..N:
    # value = (2A/H) * (X(I) - X(I-1)) / (cos(2π*I*H) - X(I))
    # RANGES = 1.0: lcon=0, ucon=1
    ExaModels.@add_con(c,
        TWO_A_H * (x[d.I + 1] - x[d.I]) / (ct[d.I] - x[d.I + 1])
        for d in [(I = I,) for I in 1:N];
        lcon = T(0), ucon = T(1))

    return ExaModels.ExaModel(c; kwargs...)
end

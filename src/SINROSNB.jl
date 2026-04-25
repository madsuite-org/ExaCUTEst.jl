# SINROSNB: Sine-Rosenbrock variation (N=1000 variables)
# Rosenbrock-like problem where squares are replaced by sines
# Source: Bouriacha, Gould, Toint, October 1993
# SIF classification: OQR2-AN-V-V
#
# GROUP TYPE SIN: F(v) = sin(v) + 1  (with scale σ: contribution = (sin(v)+1)/σ)
# ELEMENT ETYPE(V1): F = V1^2
#
# Objective groups (N type):
#   SQ1 (SIN, no scale): linear x[1] coeff 1.0, const RHS=1-3π/2
#     v = x[1] - (1 - 3π/2) = x[1] - 1 + 3π/2;  contrib = sin(v) + 1
#   SQ(I) I=2..N (SIN, scale=0.01): linear x[I] coeff -1.0, element XSQ(I-1)=x[I-1]^2, const RHS=-3π/2
#     v = x[I-1]^2 - x[I] + 3π/2;  contrib = (sin(v) + 1) / 0.01
#
# Constraint groups (G type, identity):
#   C(I) I=2..N: linear x[I] coeff -1.0, element XSQ(I-1), const RHS=-π, RANGE=2π
#     v = x[I-1]^2 - x[I] + π;  lcon=0, ucon=2π
#
# Bounds: all free, x[1] in [-π, π]
# Start: x[1]=-1.0, x[I]=10 for I=2..N
function sinrosnb_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    pi_val = T(3.1415926535)
    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    lvar[1] = -pi_val
    uvar[1] = pi_val
    start = [T(-1.0); fill(T(10.0), N - 1)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)
    # SQ1: no scale → contribution = sin(v) + 1 where v = x[1] - 1 + 3π/2
    ExaModels.@add_obj(c, sin(x[1] - T(1) + T(1.5) * pi_val) + T(1))
    # SQ(I) for I=2..N: scale σ=0.01 → contribution = (sin(v) + 1) / 0.01
    # where v = x[I-1]^2 - x[I] + 3π/2
    ExaModels.@add_obj(c, (sin(x[i-1]^2 - x[i] + T(1.5) * pi_val) + T(1)) / T(0.01) for i in 2:N)
    # C(I) for I=2..N: x[I-1]^2 - x[I] + π, lcon=0, ucon=2π
    ExaModels.@add_con(c, x[i-1]^2 - x[i] + pi_val for i in 2:N;
                       lcon = T(0), ucon = T(2) * pi_val)
    return ExaModels.ExaModel(c; kwargs...)
end

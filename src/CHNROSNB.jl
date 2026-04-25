# CHNROSNB: Chained Rosenbrock function (Toint), N=50 variables, unconstrained
# minimize sum_{i=2}^{N} [ 16*ALPH(i)^2 * (x[i-1] - x[i]^2)^2 + (x[i] - 1)^2 ]
#
# SIF structure:
#   Groups SQ(I): linear x[I-1], element ELA(I)=-x[I]^2 → v=x[I-1]-x[I]^2
#                 L2 group type, scale σ=1/(16*ALPH(I)^2)
#                 contribution = v^2/σ = 16*ALPH(I)^2 * (x[I-1]-x[I]^2)^2
#   Groups B(I):  linear x[I], constant 1 → v=x[I]-1
#                 L2 group type, no scale → contribution = (x[I]-1)^2
#   Bounds: FR DEFAULT → all variables free
#   Start: all -1.0
function chnrosnb_model(; T = Float64, backend = nothing, kwargs...)
    N = 50
    # ALPH(1)..ALPH(50) from SIF
    alph = T[
        1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
        1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
        1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
        1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
        2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50,
    ]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                       start = fill(T(-1.0), N))
    # Precompute 16*ALPH(I)^2 as parameters (indexed I=2..N → array index 1..N-1)
    ExaModels.@add_par(c, scl, T[16 * alph[i]^2 for i in 2:N])
    # SQ(I) groups: 16*ALPH(I)^2 * (x[I-1] - x[I]^2)^2
    ExaModels.@add_obj(c, scl[i] * (x[i] - x[i+1]^2)^2 for i in 1:N-1)
    # B(I) groups: (x[I] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end

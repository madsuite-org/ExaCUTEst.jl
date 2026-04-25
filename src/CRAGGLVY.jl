# CRAGGLVY: Extended Cragg and Levy problem (N=5000 vars, unconstrained)
# Source: Toint (1983), problem 32; Buckley#18
# SIF: M groups sets of 5 objective terms each; N = 2M+2 variables, all free.
# Default M = 2499 (N = 5000).
#
# For each I = 1..M:
#   A(I): L4,  v = exp(x[2I-1]) - x[2I]                              → v^4
#   B(I): L6, SCALE=0.01, v = x[2I] - x[2I+1]                        → v^6/0.01 = 100*v^6
#   C(I): L4,  v = (x[2I+1]-x[2I+2]) + tan(x[2I+1]-x[2I+2])          → v^4
#   D(I): L8,  v = x[2I-1]                                             → v^8
#   F(I): L2,  v = x[2I+2] - 1  (CONSTANT RHS=1 for F groups)         → v^2
#
# Variables: all free (FR DEFAULT). Start: x=2.0, x[1]=1.0.
function cragglvy_model(M = 2499; T = Float64, backend = nothing, kwargs...)
    N = 2M + 2
    start = fill(T(2.0), N)
    start[1] = T(1.0)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # A(I): L4, v = exp(x[2I-1]) - x[2I]
    ExaModels.@add_obj(c, (exp(x[2i-1]) - x[2i])^4 for i in 1:M)
    # B(I): L6, SCALE=0.01 → v^6/0.01 = 100*v^6
    ExaModels.@add_obj(c, T(100) * (x[2i] - x[2i+1])^6 for i in 1:M)
    # C(I): L4, v = (x[2I+1]-x[2I+2]) + tan(x[2I+1]-x[2I+2])
    ExaModels.@add_obj(c, let u = x[2i+1] - x[2i+2]; (u + tan(u))^4 end for i in 1:M)
    # D(I): L8, v = x[2I-1]
    ExaModels.@add_obj(c, x[2i-1]^8 for i in 1:M)
    # F(I): L2, v = x[2I+2] - 1
    ExaModels.@add_obj(c, (x[2i+2] - T(1))^2 for i in 1:M)
    return ExaModels.ExaModel(c; kwargs...)
end

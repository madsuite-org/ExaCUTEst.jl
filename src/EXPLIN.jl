# EXPLIN: Problem involving exponential and linear terms (N=1200, M=100)
# minimize sum_{i=1}^{N} (i - 10)*x[i] + sum_{i=1}^{M} exp(0.1*x[i]*x[i+1])
#
# SIF: single OBJ group (identity type), no constant RHS
#   Linear part: coefficient C = RI * (-10.0) = (i - 10) * ... wait:
#     RI RI I => RI = i (real); RM C RI -10.0 => C = RI * (-10.0) = -10*i
#     ZN OBJ X(I) C => linear coefficient of x[i] is C = -10*i
#   But wait: RM C RI -10.0 reads as C = RI + (-10.0) or C = RI * (-10.0)?
#   In SIF: RM means param1 = param2 * scalar, so C = i * (-10.0) = -10*i
#   Hmm but that gives -10, -20, ..., -12000 which seems too negative.
#   Actually checking SIF manual: RM p1 p2 v means p1 = p2 * v. So C = RI * (-10.0) = -10*i.
#   Wait, re-reading: RM C RI - 10.0 - the "- 10.0" has a minus sign!
#   In the SIF file: " RM C         RI         - 10.0" -> the space before 10 suggests it is "-10.0"
#   So C = RI * (-10.0)? But checking RA/RM semantics again:
#   RA p1 p2 v -> p1 = p2 + v
#   RM p1 p2 v -> p1 = p2 * v
#   So RM C RI -10.0 -> C = RI * (-10.0) = -10*i
#
#   Element E(I): exp(0.1 * x[I] * x[I+1]) for I=1..M
#   Bounds: XU DEFAULT 10.0 -> uvar=10, lvar=0 (default), start=0
function explin_model(; T = Float64, backend = nothing, kwargs...)
    N = 1200
    M = 100
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(10), start = T(0))
    ExaModels.@add_par(c, θ, T[-10 * i for i in 1:N])
    # Linear part: sum_{i=1}^{N} (-10*i) * x[i]
    ExaModels.@add_obj(c, θ[i] * x[i] for i in 1:N)
    # Exponential part: sum_{i=1}^{M} exp(0.1 * x[i] * x[i+1])
    ExaModels.@add_obj(c, exp(T(0.1) * x[i] * x[i+1]) for i in 1:M)
    return ExaModels.ExaModel(c; kwargs...)
end

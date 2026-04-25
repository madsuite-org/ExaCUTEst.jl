# COSINE: sum of cosines of quadratic expressions (N=10000 variables, unconstrained)
# minimize sum_{i=1}^{N-1} cos(x[i]^2 - 0.5*x[i+1])
# Source: N. Gould, Jan 1996
# SIF: N groups G(I) with COS group type; element SQ: X(I)^2; linear -0.5*X(I+1)
#   Group value: v = X(I)^2 + (-0.5)*X(I+1); contribution = cos(v)
# Default N = 10000 (active IE N line)
function cosine_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c, cos(x[i]^2 - T(0.5) * x[i+1]) for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end

# SROSENBR: Separable Rosenbrock (N=5000 variables, N/2=2500 pairs, unconstrained)
# minimize sum_{i=1}^{N/2} [100*(x[2i] - x[2i-1]^2)^2 + (x[2i-1] - 1)^2]
# All variables free; start: x[2i-1]=-1.2, x[2i]=1.0
# Default N/2 = 2500 (N = 5000), matching SIF default IE N/2 = 2500
#
# SIF structure:
#   GA(I) groups: L2 type, SCALE = 1/ALPHA = 1/100
#     linear part X(2I) + element MSQ(X(2I-1)) = -X(2I-1)^2
#     v = X(2I) - X(2I-1)^2
#     contribution = v^2 / (1/100) = 100*(X(2I) - X(2I-1)^2)^2
#   GB(I) groups: L2 type, no scale
#     linear part X(2I-1), constant = 1.0
#     v = X(2I-1) - 1.0
#     contribution = (X(2I-1) - 1)^2
function srosenbr_model(NH = 2500; T = Float64, backend = nothing, kwargs...)
    N  = 2 * NH
    # Starting point: x[2i-1] = -1.2, x[2i] = 1.0
    x0 = Vector{T}(undef, N)
    for i in 1:NH
        x0[2i-1] = T(-1.2)
        x0[2i]   = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    # GA(I): 100 * (x[2i] - x[2i-1]^2)^2
    ExaModels.@add_obj(c, T(100) * (x[2i] - x[2i-1]^2)^2 for i in 1:NH)
    # GB(I): (x[2i-1] - 1)^2
    ExaModels.@add_obj(c, (x[2i-1] - T(1))^2 for i in 1:NH)
    return ExaModels.ExaModel(c; kwargs...)
end

# WOODS: Extended Woods problem (4*NS variables, unconstrained)
# minimize sum_{i=1}^{NS} [
#   (x[4i-2] - x[4i-3]^2)^2 / 0.01          [A(i): L2, scale=0.01]
#   + (1 - x[4i-3])^2                         [B(i): L2, v = -x[4i-3] + 1]
#   + 90*(x[4i] - x[4i-1]^2)^2               [C(i): L2, scale=1/90]
#   + (1 - x[4i-1])^2                         [D(i): L2, v = -x[4i-1] + 1]
#   + (x[4i-2] + x[4i] - 2)^2 / 0.1          [E(i): L2, scale=0.1]
#   + (x[4i-2] - x[4i])^2 / 10               [F(i): L2, scale=10]
# ]
# Default NS = 1000 → N = 4000 variables
# Source: More', Garbow, Hillstrom (1981), problem 14
# SIF element MSQ: F = -V*V  (note: sign is negative, so element adds -x^2 to linear part)
# Group A(I): linear x[J-2], element Y(I)=-x[J-3]^2 → v = x[J-2] - x[J-3]^2
# Group C(I): linear x[J],   element Z(I)=-x[J-1]^2 → v = x[J] - x[J-1]^2
function woods_model(NS = 1000; T = Float64, backend = nothing, kwargs...)
    N = 4 * NS
    # Start: for odd indices i=1,3,5,...: x[i]=-3, x[i+1]=-1
    # SIF: DI I 2 means increment by 2, so i goes 1,3,5,...,N-1
    x0 = Vector{T}(undef, N)
    for i in 1:2:N
        x0[i]   = T(-3.0)
        x0[i+1] = T(-1.0)
    end
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free (FR WOODS 'DEFAULT')
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    # Group CONST: N type, no terms → contribution = 0, skip
    # For each set i=1..NS, J=4i, indices J-3=4i-3, J-2=4i-2, J-1=4i-1, J=4i
    # A(i): scale=0.01, v = x[4i-2] + (-x[4i-3]^2) = x[4i-2] - x[4i-3]^2
    # contribution = v^2 / 0.01 = 100 * v^2
    ExaModels.@add_obj(c, (x[4i-2] - x[4i-3]^2)^2 / T(0.01) for i in 1:NS)
    # B(i): v = -x[4i-3] - (-1) = 1 - x[4i-3], contribution = v^2
    ExaModels.@add_obj(c, (T(1) - x[4i-3])^2 for i in 1:NS)
    # C(i): scale=1/90, v = x[4i] + (-x[4i-1]^2) = x[4i] - x[4i-1]^2
    # contribution = v^2 / (1/90) = 90 * v^2
    ExaModels.@add_obj(c, (x[4i] - x[4i-1]^2)^2 / (T(1) / T(90)) for i in 1:NS)
    # D(i): v = -x[4i-1] - (-1) = 1 - x[4i-1], contribution = v^2
    ExaModels.@add_obj(c, (T(1) - x[4i-1])^2 for i in 1:NS)
    # E(i): scale=0.1, v = x[4i-2] + x[4i] - 2, contribution = v^2 / 0.1
    ExaModels.@add_obj(c, (x[4i-2] + x[4i] - T(2))^2 / T(0.1) for i in 1:NS)
    # F(i): scale=10, v = x[4i-2] - x[4i], contribution = v^2 / 10
    ExaModels.@add_obj(c, (x[4i-2] - x[4i])^2 / T(10) for i in 1:NS)
    return ExaModels.ExaModel(c; kwargs...)
end

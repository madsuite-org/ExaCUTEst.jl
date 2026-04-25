# CHAINWOO: Chained Woods function (N=2*NS+2 variables, unconstrained)
# minimize CONST + sum_{i=1}^{NS} [
#   100*(x[2i] - x[2i-1]^2)^2      [A(i): L2, scale=0.01, element MSQ(x[2i-1])=-x[2i-1]^2]
#   + (1 - x[2i-1])^2              [B(i): L2, v = -x[2i-1] + 1]
#   + 90*(x[2i+2] - x[2i+1]^2)^2  [C(i): L2, scale=1/90, element MSQ(x[2i+1])=-x[2i+1]^2]
#   + (1 - x[2i+1])^2              [D(i): L2, v = -x[2i+1] + 1]
#   + 10*(x[2i] + x[2i+2] - 2)^2  [E(i): L2, scale=0.1]
#   + (x[2i] - x[2i+2])^2 / 10    [F(i): L2, scale=10]
# ]
# CONST group: L2 type (T 'DEFAULT' L2 applies), no vars, constant=1.0 → v=-1, contribution=1
# Default NS = 1999 → N = 4000 variables
# Source: Conn, Gould, Toint (1988), problem 8
# SIF element MSQ: F = -V*V (negative square added to linear part)
# Start: x[1]=-3, x[2]=-1, x[3]=-3, x[4]=-1, rest=-2
function chainwoo_model(NS = 1999; T = Float64, backend = nothing, kwargs...)
    N = 2 * NS + 2
    x0 = fill(T(-2.0), N)
    x0[1] = T(-3.0)
    x0[2] = T(-1.0)
    x0[3] = T(-3.0)
    x0[4] = T(-1.0)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free (FR CHAINWOO 'DEFAULT')
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    # CONST group: L2 type (T 'DEFAULT' L2), no variables, constant=1
    #   v = 0 - 1 = -1 → L2 contribution = v^2 / 1 = (-1)^2 = 1
    ExaModels.@add_par(c, const_val, T[1.0])
    ExaModels.@add_obj(c, const_val[i] for i in 1:1)
    # For each set i=1..NS, indices: a=2i-1, b=2i, c_=2i+1, d=2i+2
    # A(i): scale=0.01, v = x[b] + (-x[a]^2) = x[b] - x[a]^2
    #   contribution = v^2 / 0.01 = 100 * v^2
    ExaModels.@add_obj(c, T(100) * (x[2i] - x[2i-1]^2)^2 for i in 1:NS)
    # B(i): v = -x[a] - (-1) = 1 - x[a], contribution = v^2
    ExaModels.@add_obj(c, (T(1) - x[2i-1])^2 for i in 1:NS)
    # C(i): scale=1/90, v = x[d] + (-x[c_]^2) = x[d] - x[c_]^2
    #   contribution = v^2 / (1/90) = 90 * v^2
    ExaModels.@add_obj(c, T(90) * (x[2i+2] - x[2i+1]^2)^2 for i in 1:NS)
    # D(i): v = -x[c_] - (-1) = 1 - x[c_], contribution = v^2
    ExaModels.@add_obj(c, (T(1) - x[2i+1])^2 for i in 1:NS)
    # E(i): scale=0.1, v = x[b] + x[d] - 2, contribution = v^2 / 0.1 = 10 * v^2
    ExaModels.@add_obj(c, T(10) * (x[2i] + x[2i+2] - T(2))^2 for i in 1:NS)
    # F(i): scale=10, v = x[b] - x[d], contribution = v^2 / 10
    ExaModels.@add_obj(c, (x[2i] - x[2i+2])^2 / T(10) for i in 1:NS)
    return ExaModels.ExaModel(c; kwargs...)
end

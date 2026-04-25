# CLPLATEB: Clamped plate problem B (P=71 default, unconstrained)
# Same as CLPLATEA but with distributed load: W group uses DISW*sum_J X(P,J)
# where DISW = WGHT/(P-1), WGHT=-0.1
# Source: J. Nocedal (1981), Cocoyoc conference proceedings
function clplateb_model(P = 71; T = Float64, backend = nothing, kwargs...)
    N     = P * P
    HP2   = T(P * P) * T(0.5)      # = P^2 / 2
    WGHT  = T(-0.1)
    DISW  = WGHT / T(P - 1)        # distributed weight per column

    idx(i, j) = (j - 1) * P + i

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf),  N)
    for j in 1:P
        k = idx(1, j)
        lvar[k] = T(0)
        uvar[k] = T(0)
    end

    grid_pairs = [(i, j) for i in 2:P for j in 2:P]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = T(0))

    ExaModels.@add_obj(c, (x[idx(i,j)] - x[idx(i,j-1)])^2 / T(2) for (i, j) in grid_pairs)
    ExaModels.@add_obj(c, (x[idx(i,j)] - x[idx(i-1,j)])^2 / T(2) for (i, j) in grid_pairs)
    ExaModels.@add_obj(c, (x[idx(i,j)] - x[idx(i,j-1)])^4 * HP2  for (i, j) in grid_pairs)
    ExaModels.@add_obj(c, (x[idx(i,j)] - x[idx(i-1,j)])^4 * HP2  for (i, j) in grid_pairs)

    # W group: distributed load DISW * sum_{J=1}^P X(P,J)
    ExaModels.@add_obj(c, DISW * x[idx(P, j)] for j in 1:P)

    return ExaModels.ExaModel(c; kwargs...)
end

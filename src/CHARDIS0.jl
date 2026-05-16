# CHARDIS0: Distribution of equal charges on [-R,R]x[-R,R] (2D)
# NP1=1000, 2*NP1 variables x(i),y(i), no constraints
# Objective: sum_{i<j} sigma^2 / ((x_i-x_j)^2 + (y_i-y_j)^2)
# where sigma=0.01, group SCALE=0.01 -> contribution = sigma * phi(v/sigma) = sigma * 1/(v/sigma) = sigma^2/v
# v = (x_i-x_j)^2 + (y_i-y_j)^2
# Start: spiral pattern x_i = 0.5*(R*(NP1-i)/N)*cos(2pi*i/N), y similarly
function chardis0_model(NP1 = 1000; T = Float64, backend = nothing, kwargs...)
    R = T(10.0)
    N = NP1 - 1
    sigma = T(0.01)
    sigma2 = sigma * sigma

    # Variable indexing: x(i) = 2*(i-1)+1, y(i) = 2*(i-1)+2
    x_idx(i) = 2 * (i - 1) + 1
    y_idx(i) = 2 * (i - 1) + 2
    NV = 2 * NP1

    lvar = fill(-R, NV)
    uvar = fill(R, NV)

    # Start point: spiral
    start = zeros(T, NV)
    two_pi_over_N = T(2) * T(pi) / T(N)
    R_over_N = R / T(N)
    for i in 1:NP1
        phi = two_pi_over_N * T(i)
        ri = R_over_N * T(NP1 - i)
        start[x_idx(i)] = T(0.5) * ri * cos(phi)
        start[y_idx(i)] = T(0.5) * ri * sin(phi)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(xi = x_idx(i), yi = y_idx(i), xj = x_idx(j), yj = y_idx(j))
             for i in 1:NP1 for j in i+1:NP1]

    ExaModels.@add_obj(c,
        sigma2 / ((x[d.xi] - x[d.xj])^2 + (x[d.yi] - x[d.yj])^2)
        for d in inner)

    return ExaModels.ExaModel(c; kwargs...)
end

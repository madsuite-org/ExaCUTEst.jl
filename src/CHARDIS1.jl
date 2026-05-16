# CHARDIS1: Distribution of charges on a round plate (2D)
# NP1=1000, 2*NP1 variables: x(1)=R fixed, y(1)=0 fixed, x(i),y(i) free for i=2..NP1
# Objective: sum_{i<j} 1/((x_i-x_j)^2 + (y_i-y_j)^2)  (no scale, sigma=1)
# Constraints (L-type): x(i)^2 + y(i)^2 - R^2 <= 0, i=2..NP1
# Start: spiral pattern (similar to CHARDIS0 but with R=1.0)
function chardis1_model(NP1 = 1000; T = Float64, backend = nothing, kwargs...)
    R = T(1.0)
    R2 = R * R
    N = NP1 - 1

    # Variable indexing: x(i) = 2*(i-1)+1, y(i) = 2*(i-1)+2
    x_idx(i) = 2 * (i - 1) + 1
    y_idx(i) = 2 * (i - 1) + 2
    NV = 2 * NP1

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    # x(1) = R, y(1) = 0 fixed
    lvar[x_idx(1)] = R; uvar[x_idx(1)] = R
    lvar[y_idx(1)] = T(0); uvar[y_idx(1)] = T(0)

    # Start: for i=2..NP1 use spiral starting point
    start = zeros(T, NV)
    start[x_idx(1)] = R
    two_pi_over_N = T(2) * T(pi) / T(N)
    R_over_N = R / T(N)
    for i in 2:NP1
        phi = two_pi_over_N * T(i - 1)
        ri = R_over_N * T(NP1 - i)
        start[x_idx(i)] = ri * cos(phi)
        start[y_idx(i)] = ri * sin(phi)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(xi = x_idx(i), yi = y_idx(i), xj = x_idx(j), yj = y_idx(j))
             for i in 1:NP1 for j in i+1:NP1]

    ExaModels.@add_obj(c,
        T(1) / ((x[d.xi] - x[d.xj])^2 + (x[d.yi] - x[d.yj])^2)
        for d in inner)

    # Constraints: x(i)^2 + y(i)^2 - R^2 <= 0 for i=2..NP1
    inner_con = [(xi = x_idx(i), yi = y_idx(i)) for i in 2:NP1]
    ExaModels.@add_con(c,
        x[d.xi]^2 + x[d.yi]^2 - R2
        for d in inner_con; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end

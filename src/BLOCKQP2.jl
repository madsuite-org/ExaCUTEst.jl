# BLOCKQP2: Non-convex quadratic program with block structure (Y coefficient -1 in constraints)
# N=5000 equality constraints, B=10 block variables (defaults)
# Variables: X(I), Y(I) for I=1..N and Z(J) for J=1..B
# Objective: sum X(I)*Y(I) + sum Z(J)^2
# Equality constraints: X(I) - Y(I) + sum(Z(J)) = B  for I=1..N
# Inequality constraint: sum_{I=1}^N (X(I)+Y(I)) + sum Z(J) >= B+1
# Bounds: -1<=X,Y<=1; 0<=Z<=2; start=0.5
# Source: Nick Gould, June 1994
function blockqp2_model(N = 5000, B = 10; T = Float64, backend = nothing, kwargs...)
    RB  = T(B)
    RB1 = T(B + 1)

    NV = 2 * N + B
    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    for i in 1:N
        lvar[i]     = T(-1); uvar[i]     = T(1)
        lvar[N + i] = T(-1); uvar[N + i] = T(1)
    end
    for j in 1:B
        lvar[2*N + j] = T(0); uvar[2*N + j] = T(2)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0.5))

    xy_data = [(i = i,) for i in 1:N]
    ExaModels.@add_obj(c, x[d.i] * x[N + d.i] for d in xy_data)
    z_data = [(j = j,) for j in 1:B]
    ExaModels.@add_obj(c, T(1//2) * x[2*N + d.j]^2 for d in z_data)

    # Equality constraints: x[I] - x[N+I] + sum_J x[2N+J] = B  for I=1..N
    ExaModels.@add_con(c,
        x[d.i] - x[N + d.i] + sum(x[2*N + j] for j in 1:B) - RB
        for d in xy_data)

    # Inequality constraint: sum_I (x[I]+x[N+I]) + sum_J x[2N+J] >= B+1
    ExaModels.@add_con(c,
        sum(x[i] + x[N+i] for i in 1:N) + sum(x[2*N+j] for j in 1:B) - RB1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

# BLOCKQP3: Objective coefficient I/N for X(I)*Y(I); constraints same as BLOCKQP1
# Source: Nick Gould, June 1994
function blockqp3_model(N = 5000, B = 10; T = Float64, backend = nothing, kwargs...)
    RB  = T(B)
    RB1 = T(B + 1)

    NV = 2 * N + B
    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    for i in 1:N
        lvar[i]     = T(-1); uvar[i]     = T(1)
        lvar[N + i] = T(-1); uvar[N + i] = T(1)
    end
    for j in 1:B
        lvar[2*N + j] = T(0); uvar[2*N + j] = T(2)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0.5))

    # Objective: sum_I (I/N)*X(I)*Y(I) + sum_J Z(J)^2
    # Coefficient I/N for each PROD element P(I)
    xy_data = [(i = i, c = T(i) / T(N)) for i in 1:N]
    ExaModels.@add_obj(c, d.c * x[d.i] * x[N + d.i] for d in xy_data)
    z_data = [(j = j,) for j in 1:B]
    ExaModels.@add_obj(c, T(1//2) * x[2*N + d.j]^2 for d in z_data)

    # Equality constraints: x[I] + x[N+I] + sum_J x[2N+J] = B  (same as BLOCKQP1)
    xy_idx = [(i = i,) for i in 1:N]
    ExaModels.@add_con(c,
        x[d.i] + x[N + d.i] + sum(x[2*N + j] for j in 1:B) - RB
        for d in xy_idx)

    ExaModels.@add_con(c,
        sum(x[i] + x[N+i] for i in 1:N) + sum(x[2*N+j] for j in 1:B) - RB1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

# BLOCKQP4: BLOCKQP2 constraints (Y coeff -1) + BLOCKQP3 objective (I/N coefficient)
# Source: Nick Gould, June 1994
function blockqp4_model(N = 5000, B = 10; T = Float64, backend = nothing, kwargs...)
    RB  = T(B)
    RB1 = T(B + 1)

    NV = 2 * N + B
    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    for i in 1:N
        lvar[i]     = T(-1); uvar[i]     = T(1)
        lvar[N + i] = T(-1); uvar[N + i] = T(1)
    end
    for j in 1:B
        lvar[2*N + j] = T(0); uvar[2*N + j] = T(2)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0.5))

    # Objective: sum_I (I/N)*X(I)*Y(I) + sum_J Z(J)^2
    xy_data = [(i = i, c = T(i) / T(N)) for i in 1:N]
    ExaModels.@add_obj(c, d.c * x[d.i] * x[N + d.i] for d in xy_data)
    z_data = [(j = j,) for j in 1:B]
    ExaModels.@add_obj(c, T(1//2) * x[2*N + d.j]^2 for d in z_data)

    # Equality constraints: x[I] - x[N+I] + sum_J x[2N+J] = B  (BLOCKQP2 form)
    xy_idx = [(i = i,) for i in 1:N]
    ExaModels.@add_con(c,
        x[d.i] - x[N + d.i] + sum(x[2*N + j] for j in 1:B) - RB
        for d in xy_idx)

    ExaModels.@add_con(c,
        sum(x[i] + x[N+i] for i in 1:N) + sum(x[2*N+j] for j in 1:B) - RB1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

# BLOCKQP5: Same as BLOCKQP3 but Z bounded [0,1] (not [0,2])
# Source: Nick Gould, June 1994
function blockqp5_model(N = 5000, B = 10; T = Float64, backend = nothing, kwargs...)
    RB  = T(B)
    RB1 = T(B + 1)

    NV = 2 * N + B
    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    for i in 1:N
        lvar[i]     = T(-1); uvar[i]     = T(1)
        lvar[N + i] = T(-1); uvar[N + i] = T(1)
    end
    for j in 1:B
        lvar[2*N + j] = T(0); uvar[2*N + j] = T(1)   # Z bounded [0,1]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0.5))

    # Objective: sum_I (I/N)*X(I)*Y(I) + sum_J Z(J)^2
    xy_data = [(i = i, c = T(i) / T(N)) for i in 1:N]
    ExaModels.@add_obj(c, d.c * x[d.i] * x[N + d.i] for d in xy_data)
    z_data = [(j = j,) for j in 1:B]
    ExaModels.@add_obj(c, T(1//2) * x[2*N + d.j]^2 for d in z_data)

    # Equality constraints: x[I] + x[N+I] + sum_J x[2N+J] = B  (same as BLOCKQP1/3)
    xy_idx = [(i = i,) for i in 1:N]
    ExaModels.@add_con(c,
        x[d.i] + x[N + d.i] + sum(x[2*N + j] for j in 1:B) - RB
        for d in xy_idx)

    ExaModels.@add_con(c,
        sum(x[i] + x[N+i] for i in 1:N) + sum(x[2*N+j] for j in 1:B) - RB1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

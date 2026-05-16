# BLOCKQP1: Non-convex quadratic program with block structure
# N=5000 equality constraints, B=10 block variables (defaults)
# Variables: X(I), Y(I) for I=1..N and Z(J) for J=1..B
# Objective: sum_{I=1}^N X(I)*Y(I) + sum_{J=1}^B Z(J)^2
# Equality constraints: X(I) + Y(I) + sum_{J=1}^B Z(J) = B  for I=1..N
# Inequality constraint: sum_{I=1}^N (X(I)+Y(I)) + sum_{J=1}^B Z(J) >= B+1
# Bounds: -1 <= X(I), Y(I) <= 1;  0 <= Z(J) <= 2
# Start: all = 0.5
# Source: Nick Gould, June 1994
function blockqp1_model(N = 5000, B = 10; T = Float64, backend = nothing, kwargs...)
    RB = T(B)
    RB1 = T(B + 1)

    NV = 2 * N + B   # total variables: X(1..N), Y(1..N), Z(1..B)
    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    # X(I): indices 1..N
    for i in 1:N
        lvar[i]   = T(-1)
        uvar[i]   = T(1)
        lvar[N+i] = T(-1)
        uvar[N+i] = T(1)
    end
    # Z(J): indices 2N+1..2N+B
    for j in 1:B
        lvar[2*N+j] = T(0)
        uvar[2*N+j] = T(2)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0.5))

    # Objective: sum_I x[I]*x[N+I] + sum_J x[2N+J]^2
    xy_data = [(i = i,) for i in 1:N]
    ExaModels.@add_obj(c, x[d.i] * x[N + d.i] for d in xy_data)
    z_data = [(j = j,) for j in 1:B]
    ExaModels.@add_obj(c, x[2*N + d.j]^2 for d in z_data)

    # Equality constraints: x[I] + x[N+I] + sum_{J=1}^B x[2N+J] = B for I=1..N
    ExaModels.@add_con(c,
        x[d.i] + x[N + d.i] + sum(x[2*N + j] for j in 1:B) - RB
        for d in xy_data)

    # Inequality constraint: sum_{I=1}^N (x[I]+x[N+I]) + sum_{J=1}^B x[2N+J] >= B+1
    ExaModels.@add_con(c,
        sum(x[i] + x[N+i] for i in 1:N) + sum(x[2*N+j] for j in 1:B) - RB1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end

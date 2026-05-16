# MSQRTA: dense matrix square root problem (P=32 default, n=P^2=1024)
# Find X such that X*X = A, where A = B*B and B(I,J) = sin((k^2)) for sequential k
# Constraint G(I,J): sum_T X(I,T)*X(T,J) = A(I,J)
# All variables free; start from B(I,J) - 0.8*sin(k^2)
function msqrta_model(P = 32; T = Float64, backend = nothing, kwargs...)
    # Build matrix B(I,J) = sin(k^2) for sequential k=1..P^2
    B = zeros(T, P, P)
    k = T(0)
    for i in 1:P
        for j in 1:P
            k += T(1)
            B[i, j] = sin(k^2)
        end
    end
    B[3, 1] = T(0)  # override as in SIF: RE B3,1 0.0

    # Compute A = B*B
    A = B * B

    # Starting point: B(I,J) - 0.8*sin(k^2)
    start_vec = Vector{T}(undef, P * P)
    k = T(0)
    for j in 1:P
        for i in 1:P
            k += T(1)
            sk2 = sin(k^2)
            start_vec[(j-1)*P + i] = B[i, j] + (-T(0.8)) * sk2
        end
    end

    # Variable index: X(I,J) = x[(J-1)*P + I] (column-major)
    # Actually SIF loops: DO I 1..P, DO J 1..P, X X(I,J) → row i first = row-major
    # index = (I-1)*P + J
    idx(i, j) = (i - 1) * P + j

    # Rebuild start in row-major order
    start_vec2 = Vector{T}(undef, P * P)
    k = T(0)
    for i in 1:P
        for j in 1:P
            k += T(1)
            sk2 = sin(k^2)
            start_vec2[(i-1)*P + j] = B[i, j] + (-T(0.8)) * sk2
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, P*P; lvar = T(-Inf), uvar = T(Inf), start = start_vec2)

    # Constraints G(I,J): sum_T X(I,T)*X(T,J) - A(I,J) = 0
    con_data = [(i = i, j = j, a = T(A[i, j])) for i in 1:P for j in 1:P]
    ExaModels.@add_con(c, sum(x[idx(d.i, t)] * x[idx(t, d.j)] for t in 1:P) - d.a
                       for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end

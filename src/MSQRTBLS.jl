# MSQRTBLS: least-squares variant of MSQRTB (P=32 default)
# Group type L2: F(v) = v^2
# G(I,J): v = sum_T X(I,T)*X(T,J) - A(I,J); L2 gives (sum_T X(I,T)*X(T,J) - A(I,J))^2
function msqrtbls_model(P = 32; T = Float64, backend = nothing, kwargs...)
    B = zeros(T, P, P)
    k = T(0)
    for i in 1:P
        for j in 1:P
            k += T(1)
            B[i, j] = sin(k^2)
        end
    end
    B[3, 1] = T(0)

    A = B * B

    idx(i, j) = (i - 1) * P + j

    start_vec = Vector{T}(undef, P * P)
    k = T(0)
    for i in 1:P
        for j in 1:P
            k += T(1)
            sk2 = sin(k^2)
            start_vec[(i-1)*P + j] = B[i, j] + (-T(0.8)) * sk2
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, P*P; lvar = T(-Inf), uvar = T(Inf), start = start_vec)

    obj_data = [(i = i, j = j, a = T(A[i, j])) for i in 1:P for j in 1:P]
    ExaModels.@add_obj(c, (sum(x[idx(d.i, t)] * x[idx(t, d.j)] for t in 1:P) - d.a)^2
                       for d in obj_data)

    return ExaModels.ExaModel(c; kwargs...)
end

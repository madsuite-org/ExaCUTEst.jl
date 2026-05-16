# NONMSQRT: "non-matrix square root" problem (P=70 default, n=P^2=4900)
# Same structure as MSQRTBLS but element uses E(I,J,K) = X(I,K)*X(I,J) not X(I,K)*X(K,J)
# i.e. G(I,J) = sum_K X(I,K)*X(I,J) - A(I,J) [note: both indices have first dim = I]
# L2 group type → objective = sum_{I,J} (sum_K X(I,K)*X(I,J) - A(I,J))^2
function nonmsqrt_model(P = 70; T = Float64, backend = nothing, kwargs...)
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

    # G(I,J) = sum_K X(I,K)*X(I,J) - A(I,J): note first index stays I for both
    obj_data = [(i = i, j = j, a = T(A[i, j])) for i in 1:P for j in 1:P]
    ExaModels.@add_obj(c, (sum(x[idx(d.i, k)] * x[idx(d.i, d.j)] for k in 1:P) - d.a)^2
                       for d in obj_data)

    return ExaModels.ExaModel(c; kwargs...)
end

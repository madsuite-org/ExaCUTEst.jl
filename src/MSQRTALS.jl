# MSQRTALS: Dense matrix square root problem (Nocedal & Liu / Buckley #201)
# Source: Buckley (1989), TR 1989CS-3, problem 201 (p. 93)
# SIF: P=32 (default), N=P^2=1024 variables X(I,J), all free (FR DEFAULT)
#
# B(I,J) = sin(K^2) where K = P*(I-1)+J (sequential count 1..P^2)
# A = B*B  (matrix product: A(I,J) = sum_{K=1}^P B(I,K)*B(K,J))
#
# Groups G(I,J) with L2 type (no scale): contribution = v_{IJ}^2
#   v_{IJ} = sum_{T=1}^P X(I,T)*X(T,J) - A(I,J)
# Objective: sum_{I,J=1}^P v_{IJ}^2
#
# Starting point: X(I,J)_0 = B(I,J) + (-0.8)*B(I,J) = 0.2*B(I,J)
#   (SIF: RM -4SK2/5 SK2 -0.8 → -4SK2/5 = SK2*(-0.8) = -0.8*sin(K^2)
#    then A+ XIJ = B(I,J) + (-4SK2/5) = sin(K^2) - 0.8*sin(K^2) = 0.2*sin(K^2))
function msqrtals_model(P = 32; T = Float64, backend = nothing, kwargs...)
    N = P * P

    # Compute B(I,J) = sin(K^2) for K = P*(I-1)+J
    B = Matrix{T}(undef, P, P)
    K = 0.0
    for i in 1:P
        for j in 1:P
            K += 1.0
            B[i, j] = T(sin(K * K))
        end
    end

    # Compute A = B * B
    A = B * B

    # Flatten A for use as parameters (row-major: A_flat[P*(i-1)+j] = A[i,j])
    A_flat = T[A[i, j] for i in 1:P for j in 1:P]

    # Starting point: X(I,J)_0 = 0.2 * B(I,J)
    x0 = T[T(0.2) * B[i, j] for i in 1:P for j in 1:P]

    # Variable index: x[P*(i-1)+j] corresponds to X(I,J)
    # Group (i,j): v_{ij} = sum_{t=1}^P x[P*(i-1)+t] * x[P*(t-1)+j] - A(i,j)
    # Iterate over all (i,j) pairs
    pairs = [(i, j) for i in 1:P for j in 1:P]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    ExaModels.@add_par(c, Apar, A_flat)

    # Objective: sum_{(i,j)} (sum_{t=1}^P x[P*(i-1)+t]*x[P*(t-1)+j] - A(i,j))^2
    ExaModels.@add_obj(
        c,
        (sum(x[P*(i-1)+t] * x[P*(t-1)+j] for t in 1:P) - Apar[P*(i-1)+j])^2
        for (i, j) in pairs
    )

    return ExaModels.ExaModel(c; kwargs...)
end

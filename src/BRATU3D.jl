# BRATU3D: 3D Bratu problem on unit cube with finite differences
# P=17 (default), P^3 variables, (P-2)^3 equality constraints
# Variables U(I,J,K) for I,J,K=1..P; inner points free, boundary fixed at 0
# Constraint G(I,J,K) for I,J,K=2..P-1:
#   6*U(I,J,K) - U(I+1,J,K) - U(I-1,J,K) - U(I,J+1,K) - U(I,J-1,K)
#   - U(I,J,K+1) - U(I,J,K-1) - C*exp(U(I,J,K)) = 0
# where C = H^2 * LAMBDA, H = 1/(P-1), LAMBDA = 6.80812
# Variable indexing: U(I,J,K) → (J-1)*P^2 + (I-1)*P + K  (1-indexed)
function bratu3d_model(P = 17; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(6.80812)
    H = T(1) / T(P - 1)
    C = H * H * LAMBDA

    NV = P^3
    u_idx(i, j, k) = (j - 1) * P * P + (i - 1) * P + k

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)

    # Fix boundary (faces): U=0
    for j in 1:P, k in 1:P
        lvar[u_idx(1, j, k)] = T(0); uvar[u_idx(1, j, k)] = T(0)
        lvar[u_idx(P, j, k)] = T(0); uvar[u_idx(P, j, k)] = T(0)
    end
    for i in 2:P-1, k in 1:P
        lvar[u_idx(i, 1, k)] = T(0); uvar[u_idx(i, 1, k)] = T(0)
        lvar[u_idx(i, P, k)] = T(0); uvar[u_idx(i, P, k)] = T(0)
    end
    for i in 2:P-1, j in 2:P-1
        lvar[u_idx(i, j, 1)] = T(0); uvar[u_idx(i, j, 1)] = T(0)
        lvar[u_idx(i, j, P)] = T(0); uvar[u_idx(i, j, P)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(u_ijk   = u_idx(i, j, k),
              u_ip1jk = u_idx(i + 1, j, k),
              u_im1jk = u_idx(i - 1, j, k),
              u_ijp1k = u_idx(i, j + 1, k),
              u_ijm1k = u_idx(i, j - 1, k),
              u_ijkp1 = u_idx(i, j, k + 1),
              u_ijkm1 = u_idx(i, j, k - 1))
             for i in 2:P-1 for j in 2:P-1 for k in 2:P-1]

    ExaModels.@add_con(c,
        T(6) * x[d.u_ijk] - x[d.u_ip1jk] - x[d.u_im1jk] -
        x[d.u_ijp1k] - x[d.u_ijm1k] - x[d.u_ijkp1] - x[d.u_ijkm1] -
        C * exp(x[d.u_ijk])
        for d in inner)

    return ExaModels.ExaModel(c; kwargs...)
end

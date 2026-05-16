# CBRATU3D: Complex 3D Bratu problem on unit cube
# P=12 (default n=3456), 2*P^3 variables U(I,J,K) and X(I,J,K)
# 2*(P-2)^3 equality constraints
# G(I,J,K): 6*U - neighbors_U - C*exp(U)*cos(X) = 0
# F(I,J,K): 6*X - neighbors_X - C*exp(U)*sin(X) = 0
# C = H^2*LAMBDA, H = 1/(P-1), LAMBDA=6.80812
# Variable ordering: DO J DO I DO K with U first, X second in each (I,J,K)
# U(I,J,K) = 2*((J-1)*P^2 + (I-1)*P + (K-1)) + 1
# X(I,J,K) = 2*((J-1)*P^2 + (I-1)*P + (K-1)) + 2
function cbratu3d_model(P = 12; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(6.80812)
    H = T(1) / T(P - 1)
    C = H * H * LAMBDA

    NV = 2 * P^3
    u_idx(i, j, k) = 2 * ((j - 1) * P * P + (i - 1) * P + (k - 1)) + 1
    v_idx(i, j, k) = 2 * ((j - 1) * P * P + (i - 1) * P + (k - 1)) + 2

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)

    # Fix boundary
    for j in 1:P, k in 1:P
        for fn in (u_idx, v_idx)
            lvar[fn(1, j, k)] = T(0); uvar[fn(1, j, k)] = T(0)
            lvar[fn(P, j, k)] = T(0); uvar[fn(P, j, k)] = T(0)
        end
    end
    for i in 2:P-1, k in 1:P
        for fn in (u_idx, v_idx)
            lvar[fn(i, 1, k)] = T(0); uvar[fn(i, 1, k)] = T(0)
            lvar[fn(i, P, k)] = T(0); uvar[fn(i, P, k)] = T(0)
        end
    end
    for i in 2:P-1, j in 2:P-1
        for fn in (u_idx, v_idx)
            lvar[fn(i, j, 1)] = T(0); uvar[fn(i, j, 1)] = T(0)
            lvar[fn(i, j, P)] = T(0); uvar[fn(i, j, P)] = T(0)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(u_ijk   = u_idx(i, j, k),   v_ijk   = v_idx(i, j, k),
              u_ip1jk = u_idx(i + 1, j, k), u_im1jk = u_idx(i - 1, j, k),
              u_ijp1k = u_idx(i, j + 1, k), u_ijm1k = u_idx(i, j - 1, k),
              u_ijkp1 = u_idx(i, j, k + 1), u_ijkm1 = u_idx(i, j, k - 1),
              v_ip1jk = v_idx(i + 1, j, k), v_im1jk = v_idx(i - 1, j, k),
              v_ijp1k = v_idx(i, j + 1, k), v_ijm1k = v_idx(i, j - 1, k),
              v_ijkp1 = v_idx(i, j, k + 1), v_ijkm1 = v_idx(i, j, k - 1))
             for i in 2:P-1 for j in 2:P-1 for k in 2:P-1]

    ExaModels.@add_con(c,
        T(6) * x[d.u_ijk] - x[d.u_ip1jk] - x[d.u_im1jk] -
        x[d.u_ijp1k] - x[d.u_ijm1k] - x[d.u_ijkp1] - x[d.u_ijkm1] -
        C * exp(x[d.u_ijk]) * cos(x[d.v_ijk])
        for d in inner)
    ExaModels.@add_con(c,
        T(6) * x[d.v_ijk] - x[d.v_ip1jk] - x[d.v_im1jk] -
        x[d.v_ijp1k] - x[d.v_ijm1k] - x[d.v_ijkp1] - x[d.v_ijkm1] -
        C * exp(x[d.u_ijk]) * sin(x[d.v_ijk])
        for d in inner)

    return ExaModels.ExaModel(c; kwargs...)
end

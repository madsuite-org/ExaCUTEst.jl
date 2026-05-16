# CBRATU2D: Complex 2D Bratu problem on unit square
# P=40 (default n=3200), 2*P^2 variables (U(I,J) real part, X(I,J) imaginary part)
# (P-2)^2 G-constraints + (P-2)^2 F-constraints = 2*(P-2)^2 total constraints
# G(I,J): 4*U(I,J) - U(I+1,J) - U(I-1,J) - U(I,J+1) - U(I,J-1) - C*exp(U)*cos(X) = 0
# F(I,J): 4*X(I,J) - X(I+1,J) - X(I-1,J) - X(I,J+1) - X(I,J-1) - C*exp(U)*sin(X) = 0
# where C = H^2*LAMBDA, H = 1/(P-1), LAMBDA=5
# Boundary: U=X=0 on all edges; start=0
# Variable indexing (loop DO J DO I): U(I,J)=2*(J-1)*P+2*(I-1)+1, X(I,J)=2*(J-1)*P+2*(I-1)+2
function cbratu2d_model(P = 40; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(5.0)
    H = T(1) / T(P - 1)
    C = H * H * LAMBDA

    NV = 2 * P * P
    u_idx(i, j) = 2 * ((j - 1) * P + (i - 1)) + 1
    v_idx(i, j) = 2 * ((j - 1) * P + (i - 1)) + 2

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)

    # Fix boundary
    for j in 1:P
        lvar[u_idx(1, j)] = T(0); uvar[u_idx(1, j)] = T(0)
        lvar[u_idx(P, j)] = T(0); uvar[u_idx(P, j)] = T(0)
        lvar[v_idx(1, j)] = T(0); uvar[v_idx(1, j)] = T(0)
        lvar[v_idx(P, j)] = T(0); uvar[v_idx(P, j)] = T(0)
    end
    for i in 2:P-1
        lvar[u_idx(i, 1)] = T(0); uvar[u_idx(i, 1)] = T(0)
        lvar[u_idx(i, P)] = T(0); uvar[u_idx(i, P)] = T(0)
        lvar[v_idx(i, 1)] = T(0); uvar[v_idx(i, 1)] = T(0)
        lvar[v_idx(i, P)] = T(0); uvar[v_idx(i, P)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(u_ij   = u_idx(i, j),   v_ij   = v_idx(i, j),
              u_ip1j = u_idx(i + 1, j), u_im1j = u_idx(i - 1, j),
              u_ijp1 = u_idx(i, j + 1), u_ijm1 = u_idx(i, j - 1),
              v_ip1j = v_idx(i + 1, j), v_im1j = v_idx(i - 1, j),
              v_ijp1 = v_idx(i, j + 1), v_ijm1 = v_idx(i, j - 1))
             for i in 2:P-1 for j in 2:P-1]

    # G(I,J): Laplace(U) - C*exp(U)*cos(X) = 0
    ExaModels.@add_con(c,
        T(4) * x[d.u_ij] - x[d.u_ip1j] - x[d.u_im1j] - x[d.u_ijp1] - x[d.u_ijm1] -
        C * exp(x[d.u_ij]) * cos(x[d.v_ij])
        for d in inner)
    # F(I,J): Laplace(X) - C*exp(U)*sin(X) = 0
    ExaModels.@add_con(c,
        T(4) * x[d.v_ij] - x[d.v_ip1j] - x[d.v_im1j] - x[d.v_ijp1] - x[d.v_ijm1] -
        C * exp(x[d.u_ij]) * sin(x[d.v_ij])
        for d in inner)

    return ExaModels.ExaModel(c; kwargs...)
end

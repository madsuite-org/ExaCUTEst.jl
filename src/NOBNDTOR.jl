# NOBNDTOR: elastic torsion problem, Q=37 default (P=2Q=74, n=P^2=5476)
# Variables: X(I,J) for J=1..P, I=1..P
# Groups (N type, identity): G(I,J) with linear coeff LC = -C*H^2
# Elements: 0.25*(X(I+1,J)-X(I,J))^2 etc. (4 neighbors per interior node)
# Boundary: X(1,J)=X(P,J)=X(I,1)=X(I,P)=0 (fixed)
# Bounds: interior nodes have symmetric bounds based on distance to boundary
# For node (I,J): bound = min(I-1, P-I, J-1, P-J) * H (where H = 1/(P-1))
# Start: upper bound value
function nobndtor_model(Q = 37; T = Float64, backend = nothing, kwargs...)
    P = 2 * Q
    H = T(1) / T(P - 1)
    H2 = H * H
    C = T(5)
    LC = -C * H2

    # Variable index: X(I,J) maps to (J-1)*P + I
    idx(i, j) = (j - 1) * P + i
    N = P * P

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start = fill(T(0), N)

    # Fix boundary variables to 0
    for j in 1:P
        k1 = idx(1, j); lvar[k1] = T(0); uvar[k1] = T(0)
        k2 = idx(P, j); lvar[k2] = T(0); uvar[k2] = T(0)
    end
    for i in 2:P-1
        k1 = idx(i, 1); lvar[k1] = T(0); uvar[k1] = T(0)
        k2 = idx(i, P); lvar[k2] = T(0); uvar[k2] = T(0)
    end

    # Interior bounds: ±min(i-1, P-i, j-1, P-j)*H
    for i in 2:P-1
        for j in 2:P-1
            d = min(i - 1, P - i, j - 1, P - j)
            bnd = T(d) * H
            k = idx(i, j)
            lvar[k] = -bnd
            uvar[k] = bnd
            start[k] = bnd  # start from upper bound
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective: linear term + quadratic terms for interior nodes
    interior = [(i = i, j = j, k = idx(i, j)) for i in 2:P-1 for j in 2:P-1]
    ExaModels.@add_obj(c, LC * x[d.k] for d in interior)

    # 4 neighbor differences, coefficient 0.25 each
    int4 = [(k = idx(i, j), kp1 = idx(i+1, j), km1 = idx(i-1, j),
             kj1 = idx(i, j+1), kjm1 = idx(i, j-1))
            for i in 2:P-1 for j in 2:P-1]
    ExaModels.@add_obj(c,
        T(0.25) * ((x[d.kp1] - x[d.k])^2 + (x[d.km1] - x[d.k])^2 +
                   (x[d.kj1] - x[d.k])^2 + (x[d.kjm1] - x[d.k])^2)
        for d in int4)

    return ExaModels.ExaModel(c; kwargs...)
end

function _torsion1_model(Q, C, start_from_upper; T = Float64, backend = nothing, kwargs...)
    # Quadratic elastic torsion problem (TORSION1 variant)
    # Q: half grid points, P=2Q, NV=P^2 variables X(I,J) for I,J=1..P
    # C: force constant, LC = -C*H^2 where H=1/(P-1)
    # Boundary: X(1,J), X(P,J), X(I,1), X(I,P) fixed at 0
    # Interior bounds: ±min(I-1, P-I, J-1, P-J)*H
    # Objective: sum_{I,J=2..P-1} [LC*X(I,J) + 0.25*sum of (X(nb)-X(I,J))^2]

    P   = 2 * Q
    P1  = P - 1
    H   = 1.0 / P1
    H2  = H * H
    LC  = -C * H2

    NV = P * P
    # Variable index: X(I,J) → (J-1)*P + I  (column-major: J outer, I inner from SIF DO J/DO I)
    idx(I, J) = (J - 1) * P + I

    lvar = zeros(T, NV)
    uvar = zeros(T, NV)
    x0   = zeros(T, NV)

    for J in 1:P
        for I in 1:P
            k = idx(I, J)
            if I == 1 || I == P || J == 1 || J == P
                # boundary: fixed at 0
                lvar[k] = T(0)
                uvar[k] = T(0)
                x0[k]   = T(0)
            else
                bnd = min(I-1, P-I, J-1, P-J) * H
                lvar[k] = T(-bnd)
                uvar[k] = T(bnd)
                x0[k]   = start_from_upper ? T(bnd) : T(0)
            end
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Linear part of objective: LC * X(I,J) for interior I,J=2..P-1
    lin_data = [(k = idx(I, J),) for J in 2:P-1 for I in 2:P-1]
    ExaModels.@add_obj(c, T(LC) * x[d.k] for d in lin_data)

    # Quadratic part: 0.25*(X(I+1,J)-X(I,J))^2 + 0.25*(X(I,J+1)-X(I,J))^2
    #               + 0.25*(X(I-1,J)-X(I,J))^2 + 0.25*(X(I,J-1)-X(I,J))^2
    quad_data = [(
        k    = idx(I, J),
        kip1 = idx(I+1, J),
        kim1 = idx(I-1, J),
        kjp1 = idx(I, J+1),
        kjm1 = idx(I, J-1),
    ) for J in 2:P-1 for I in 2:P-1]
    ExaModels.@add_obj(c,
        T(0.25) * (x[d.kip1] - x[d.k])^2
      + T(0.25) * (x[d.kjp1] - x[d.k])^2
      + T(0.25) * (x[d.kim1] - x[d.k])^2
      + T(0.25) * (x[d.kjm1] - x[d.k])^2
        for d in quad_data)

    return ExaModels.ExaModel(c; kwargs...)
end

torsion1_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 5.0, true;  T = T, backend = backend, kwargs...)
torsion2_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 5.0, false; T = T, backend = backend, kwargs...)
torsion3_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 10.0, true;  T = T, backend = backend, kwargs...)
torsion4_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 10.0, false; T = T, backend = backend, kwargs...)
torsion5_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 20.0, true;  T = T, backend = backend, kwargs...)
torsion6_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsion1_model(Q, 20.0, false; T = T, backend = backend, kwargs...)

function _torsiona_model(Q, C, start_from_upper; T = Float64, backend = nothing, kwargs...)
    # Quadratic elastic torsion problem (TORSIONA/MINPACK-2 variant)
    # Same bounds/variables as TORSION1 variant; different group decomposition
    # GR(I,J) for I,J=1..P-1: 0.25*(X(I+1,J)-X(I,J))^2 + 0.25*(X(I,J+1)-X(I,J))^2
    # GL(I,J) for I,J=2..P:   0.25*(X(I-1,J)-X(I,J))^2 + 0.25*(X(I,J-1)-X(I,J))^2
    # G(I,J) for I,J=2..P-1: LC*X(I,J)  (linear only)
    # Total objective = sum of all above (equivalent to TORSION1)

    P   = 2 * Q
    P1  = P - 1
    H   = 1.0 / P1
    H2  = H * H
    LC  = -C * H2

    NV = P * P
    idx(I, J) = (J - 1) * P + I

    lvar = zeros(T, NV)
    uvar = zeros(T, NV)
    x0   = zeros(T, NV)

    for J in 1:P
        for I in 1:P
            k = idx(I, J)
            if I == 1 || I == P || J == 1 || J == P
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

    # Linear part: G(I,J) for I,J=2..P-1: LC*X(I,J)
    lin_data = [(k = idx(I, J),) for J in 2:P-1 for I in 2:P-1]
    ExaModels.@add_obj(c, T(LC) * x[d.k] for d in lin_data)

    # GR(I,J) for I=1..P-1, J=1..P-1: A(I,J)=(X(I+1,J)-X(I,J))^2 * 0.25
    #                                   B(I,J)=(X(I,J+1)-X(I,J))^2 * 0.25
    gr_data = [(k = idx(I, J), kip1 = idx(I+1, J), kjp1 = idx(I, J+1))
               for J in 1:P-1 for I in 1:P-1]
    ExaModels.@add_obj(c,
        T(0.25) * (x[d.kip1] - x[d.k])^2 + T(0.25) * (x[d.kjp1] - x[d.k])^2
        for d in gr_data)

    # GL(I,J) for I=2..P, J=2..P: C(I,J)=(X(I-1,J)-X(I,J))^2 * 0.25
    #                               D(I,J)=(X(I,J-1)-X(I,J))^2 * 0.25
    gl_data = [(k = idx(I, J), kim1 = idx(I-1, J), kjm1 = idx(I, J-1))
               for J in 2:P for I in 2:P]
    ExaModels.@add_obj(c,
        T(0.25) * (x[d.kim1] - x[d.k])^2 + T(0.25) * (x[d.kjm1] - x[d.k])^2
        for d in gl_data)

    return ExaModels.ExaModel(c; kwargs...)
end

torsiona_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 5.0, true;  T = T, backend = backend, kwargs...)
torsionb_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 5.0, false; T = T, backend = backend, kwargs...)
torsionc_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 10.0, true;  T = T, backend = backend, kwargs...)
torsiond_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 10.0, false; T = T, backend = backend, kwargs...)
torsione_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 20.0, true;  T = T, backend = backend, kwargs...)
torsionf_model(Q = 37; T = Float64, backend = nothing, kwargs...) =
    _torsiona_model(Q, 20.0, false; T = T, backend = backend, kwargs...)

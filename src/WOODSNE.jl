function woodsne_model(NS = 1000; T = Float64, backend = nothing, kwargs...)
    # WOODSNE: Extended Woods nonlinear equations, NS sets, N=4*NS variables
    # MSQ element: F = -V^2
    # Groups per set I (J=4I, indices: J-3,J-2,J-1,J):
    #   A(I) (E): X(J-2) - X(J-3)^2 = 0, SCALE=0.1 → (X(J-2) - X(J-3)^2)/0.1 = 0
    #   B(I) (E): -X(J-3) - (-1) = 0  → X(J-3) = 1
    #   C(I) (E): X(J) - X(J-1)^2 = 0, SCALE=1/90 → (X(J) - X(J-1)^2)*90 = 0
    # Plus CONST (E): 0 - 1 = -1 = 0  (trivially infeasible constant, but exists in CUTEst)
    # Objective (N-type groups D, E, F per set):
    #   D(I): -X(J-1) - (-1) = 1 - X(J-1)  (linear, no SCALE)
    #   E(I): X(J-2) + X(J) - 2 (SCALE=0.1)... Wait:
    #     XN E(I) X(J-2) 1.0, X(J) 1.0, SCALE=0.1, RHS=2.0
    #     v_E = X(J-2) + X(J) - 2.0, obj += v_E/0.1 = (X(J-2)+X(J)-2)/0.1
    #     But identity group type, so contribution = v (not v^2)
    #   F(I): X(J-2) - X(J) (SCALE=10): v_F = X(J-2) - X(J), obj += (X(J-2)-X(J))/10
    # All vars free. Start: all -2 except X1=-3, X2=-1, X3=-3, X4=-1

    N = 4 * NS

    start = fill(T(-2), N)
    start[1] = T(-3); start[2] = T(-1); start[3] = T(-3); start[4] = T(-1)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N;
        lvar = T(-Inf), uvar = T(Inf), start = start)

    # CONST group (E type): value = -1 (no vars, no elements, RHS=1)
    ExaModels.@add_con(c, T(-1))

    # Per-set data: J=4I
    set_data = [(Jm3 = 4*I-3, Jm2 = 4*I-2, Jm1 = 4*I-1, J = 4*I) for I in 1:NS]

    # A(I) (E): (X(J-2) - X(J-3)^2) / 0.1 = 0
    ExaModels.@add_con(c, (x[d.Jm2] - x[d.Jm3]^2) / T(0.1) for d in set_data)

    # B(I) (E): -X(J-3) - (-1) = 1 - X(J-3) = 0
    ExaModels.@add_con(c, T(1) - x[d.Jm3] for d in set_data)

    # C(I) (E): (X(J) - X(J-1)^2) * 90 = 0
    ExaModels.@add_con(c, (x[d.J] - x[d.Jm1]^2) * T(90) for d in set_data)

    # D(I) (N): 1 - X(J-1)  [linear, identity group, no scale]
    ExaModels.@add_obj(c, T(1) - x[d.Jm1] for d in set_data)

    # E(I) (N): (X(J-2) + X(J) - 2) / 0.1
    ExaModels.@add_obj(c, (x[d.Jm2] + x[d.J] - T(2)) / T(0.1) for d in set_data)

    # F(I) (N): (X(J-2) - X(J)) / 10
    ExaModels.@add_obj(c, (x[d.Jm2] - x[d.J]) / T(10) for d in set_data)

    return ExaModels.ExaModel(c; kwargs...)
end

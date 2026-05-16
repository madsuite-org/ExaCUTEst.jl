function lin_model(; T = Float64, backend = nothing, kwargs...)
    # LIN: Chemical equilibrium NRTL problem, classification OLR2-AY-4-2
    # 4 variables, 2 equality constraints
    # Variables: X(I,K) for I=1..2, K=1..2
    # Ordering: X(1,1)=x[1], X(1,2)=x[2], X(2,1)=x[3], X(2,2)=x[4]
    # Bounds: lvar=1e-12, uvar=INIT(I)=0.5 for all

    # Precomputed constants
    # TAU(1,2)=3.00498, TAU(2,1)=4.69071, TAU(I,I)=0
    # ALF(1,2)=ALF(2,1)=0.391965
    # G(i,j) = exp(-ALF(i,j)*TAU(i,j)), G(i,i)=1
    # M(i,j) = G(i,j)*TAU(i,j)
    G12 = T(exp(-0.391965 * 3.00498))   # ≈ 0.30794103
    G21 = T(exp(-0.391965 * 4.69071))   # ≈ 0.15904086
    M12 = T(G12 * 3.00498)              # ≈ 0.92535663
    M21 = T(G21 * 4.69071)              # ≈ 0.74601454
    # G11=G22=1, M11=M22=0

    lvar = fill(T(1e-12), 4)
    uvar = T[0.5, 0.5, 0.5, 0.5]
    x0   = T[0.5, 0.0, 0.0, 0.5]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = lvar, uvar = uvar, start = x0)

    # Objective:
    # For each phase K (1,2):
    #   XTAUG1(Y1=X(1,K), Y2=X(2,K)) = Y1*(Y1*M11/INSUM1 + Y2*M12/INSUM2)
    #     where INSUM1=Y1*G11+Y2*G21=Y1+Y2*G21, INSUM2=Y1*G12+Y2*G22=Y1*G12+Y2
    #     M11=0, so = Y1 * (0 + Y2*M12/INSUM2) = Y1*Y2*M12/(Y1*G12+Y2)
    #   XTAUG2(Y1=X(1,K), Y2=X(2,K)) = Y2*(Y1*M21/INSUM1 + Y2*M22/INSUM2)
    #     M22=0, so = Y2*Y1*M21/(Y1+Y2*G21)
    #   For I=1..2:
    #     B(I,K) - C(I,K) = X(I,K)*log(X(I,K)) - X(I,K)*log(X(1,K)+X(2,K))
    #                      = X(I,K)*(log(X(I,K)) - log(X(1,K)+X(2,K)))
    #                      = X(I,K)*log(X(I,K)/(X(1,K)+X(2,K)))

    # K=1: Y1=x[1], Y2=x[3]
    ExaModels.@add_obj(c, x[1]*x[3]*M12/(x[1]*G12+x[3]))  # XTAUG1 K=1
    ExaModels.@add_obj(c, x[3]*x[1]*M21/(x[1]+x[3]*G21))  # XTAUG2 K=1
    ExaModels.@add_obj(c, x[1]*log(x[1]) - x[1]*log(x[1]+x[3]))  # B(1,1)-C(1,1)
    ExaModels.@add_obj(c, x[3]*log(x[3]) - x[3]*log(x[1]+x[3]))  # B(2,1)-C(2,1)

    # K=2: Y1=x[2], Y2=x[4]
    ExaModels.@add_obj(c, x[2]*x[4]*M12/(x[2]*G12+x[4]))  # XTAUG1 K=2
    ExaModels.@add_obj(c, x[4]*x[2]*M21/(x[2]+x[4]*G21))  # XTAUG2 K=2
    ExaModels.@add_obj(c, x[2]*log(x[2]) - x[2]*log(x[2]+x[4]))  # B(1,2)-C(1,2)
    ExaModels.@add_obj(c, x[4]*log(x[4]) - x[4]*log(x[2]+x[4]))  # B(2,2)-C(2,2)

    # Constraints: MB(I) = sum_K X(I,K) = INIT(I) = 0.5
    ExaModels.@add_con(c, x[1] + x[2] - T(0.5))  # MB(1)
    ExaModels.@add_con(c, x[3] + x[4] - T(0.5))  # MB(2)

    return ExaModels.ExaModel(c; kwargs...)
end

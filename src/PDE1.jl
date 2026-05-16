# PDE1: PDE-discretized LP (Mittelmann), N=299 by default
# min T  s.t. Laplacian constraints + box constraints on T +/- H*X(i,j)
# classification LLR2-AN-V-V
# Source: Hans Mittelmann (pde_1.mod), SIF input: Nick Gould, April 2012
#
# Variables: x[1]=T, x[2..nvar]=X(I,J) for I=0..N+1, J=0..N+1
# idx_X(I,J) = 2 + I*(N+2) + J  (0-based I,J from 0..N+1)
function pde1_model(N = 299; T = Float64, backend = nothing, kwargs...)
    N1 = N + 1
    A_val = T(0.01)
    G_val = T(20)
    H = T(1) / T(N1)
    GH2 = G_val * H^2
    SQRTAH = sqrt(A_val * H)

    # Total variables: 1 (T) + (N+2)^2 (X grid)
    nvar = 1 + (N1 + 1)^2

    # Variable bounds:
    # T is free (default lvar=0 but actually no bounds stated)
    # Actually: "LO PDE1 'DEFAULT' 0.0" and "UP PDE1 'DEFAULT' 3.5"
    # then boundary X(I,0), X(I,N1), X(0,I), X(N1,I) have UP 10.0
    lvar = fill(T(0), nvar)
    uvar = fill(T(3.5), nvar)
    # T variable (index 1) bounds: actually T is not bounded above in PDE formulation
    # but by default all vars 0..3.5, and T is included in that
    # Boundary X vars have uvar=10
    # X(I,0) for I=1..N
    for I in 1:N
        idx = 2 + I*(N1+1) + 0  # J=0
        uvar[idx] = T(10)
    end
    # X(I,N1) for I=1..N
    for I in 1:N
        idx = 2 + I*(N1+1) + N1
        uvar[idx] = T(10)
    end
    # X(0,I) for I=1..N
    for I in 1:N
        idx = 2 + 0*(N1+1) + I  # I-th row=0, col=I
        uvar[idx] = T(10)
    end
    # X(N1,I) for I=1..N
    for I in 1:N
        idx = 2 + N1*(N1+1) + I
        uvar[idx] = T(10)
    end

    # Helper: linear index of X(I,J)
    # idx_X(I,J) = 2 + I*(N1+1) + J  (0-based I,J)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar)

    # Objective: min T = x[1]
    ExaModels.@add_obj(c, x[1])

    # P(I,J) equality constraints for I=1..N, J=1..N:
    # 4*X(I,J) - X(I,J+1) - X(I,J-1) - X(I+1,J) - X(I-1,J) = G*H^2
    # In variable indices: x[2+I*(N1+1)+J]
    p_data = [(i=I, j=J,
               c_ij   = 2 + I*(N1+1) + J,
               c_ijp  = 2 + I*(N1+1) + (J+1),
               c_ijm  = 2 + I*(N1+1) + (J-1),
               c_ip_j = 2 + (I+1)*(N1+1) + J,
               c_im_j = 2 + (I-1)*(N1+1) + J)
              for I in 1:N for J in 1:N]
    ExaModels.@add_con(c,
        T(4)*x[d.c_ij] - x[d.c_ijp] - x[d.c_ijm] - x[d.c_ip_j] - x[d.c_im_j] - GH2
        for d in p_data)

    # Compute RHS for A/B constraints
    # YD(I,J) = -5*I*H*(I*H-1)*J*H*(J*H-1)*H
    # A(I,J) G type: T + H*X(I,J) >= YD(I,J)
    # B(I,J) L type: -T + H*X(I,J) <= YD(I,J)  (actually B uses -YD)

    ab_data = [(i=I, j=J,
                c_ij = 2 + I*(N1+1) + J,
                rhs_a = T(-5)*T(I)*H*(T(I)*H-1)*T(J)*H*(T(J)*H-1)*H,
                rhs_b = T(5)*T(I)*H*(T(I)*H-1)*T(J)*H*(T(J)*H-1)*H)
               for I in 1:N for J in 1:N]
    # A(I,J): T + H*X(I,J) >= rhs_a  → T + H*X - rhs_a >= 0
    ExaModels.@add_con(c,
        x[1] + H*x[d.c_ij] - d.rhs_a for d in ab_data;
        lcon = T(0), ucon = T(Inf))
    # B(I,J): -T + H*X(I,J) <= rhs_b → -T + H*X - rhs_b <= 0
    ExaModels.@add_con(c,
        -x[1] + H*x[d.c_ij] - d.rhs_b for d in ab_data;
        lcon = T(-Inf), ucon = T(0))

    # Boundary constraints C/D for J=0 and J=N1 (I=1..N)
    # C(I,0) G type: T + SQRTAH*X(I,0) >= 0
    # D(I,0) L type: -T + SQRTAH*X(I,0) <= 0
    cI0_data = [(c_idx = 2 + I*(N1+1) + 0,) for I in 1:N]
    ExaModels.@add_con(c, x[1] + SQRTAH*x[d.c_idx] for d in cI0_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[1] + SQRTAH*x[d.c_idx] for d in cI0_data;
                       lcon = T(-Inf), ucon = T(0))
    # C(I,N1) and D(I,N1)
    cIN1_data = [(c_idx = 2 + I*(N1+1) + N1,) for I in 1:N]
    ExaModels.@add_con(c, x[1] + SQRTAH*x[d.c_idx] for d in cIN1_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[1] + SQRTAH*x[d.c_idx] for d in cIN1_data;
                       lcon = T(-Inf), ucon = T(0))
    # E(0,I) G and F(0,I) L for I=1..N
    e0I_data = [(c_idx = 2 + 0*(N1+1) + I,) for I in 1:N]
    ExaModels.@add_con(c, x[1] + SQRTAH*x[d.c_idx] for d in e0I_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[1] + SQRTAH*x[d.c_idx] for d in e0I_data;
                       lcon = T(-Inf), ucon = T(0))
    # E(N1,I) G and F(N1,I) L for I=1..N
    eN1I_data = [(c_idx = 2 + N1*(N1+1) + I,) for I in 1:N]
    ExaModels.@add_con(c, x[1] + SQRTAH*x[d.c_idx] for d in eN1I_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[1] + SQRTAH*x[d.c_idx] for d in eN1I_data;
                       lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end

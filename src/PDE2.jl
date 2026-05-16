# PDE2: PDE-discretized LP (Mittelmann pde_2.mod), N=299 by default
# min sum T(I,J)  s.t. Laplacian + box constraints with per-cell T(I,J) variables
# classification LLR2-AN-V-V
# Source: Hans Mittelmann, SIF input: Nick Gould, April 2012
#
# Variables (in order):
#   X(I,J) for I=0..N+1, J=0..N+1 → indices 1..(N+2)^2
#   T(I,J) for I=1..N, J=0..N+1   → indices (N+2)^2+1..(N+2)^2+N*(N+2)
# idx_X(I,J) = 1 + I*(N+2) + J  (0-based)
# idx_T(I,J) = (N+2)^2 + 1 + (I-1)*(N+2) + J  (I=1..N, J=0..N+1)
function pde2_model(N = 299; T = Float64, backend = nothing, kwargs...)
    N1 = N + 1
    A_val = T(0.01)
    G_val = T(20)
    H = T(1) / T(N1)
    GH2 = G_val * H^2
    SQRTAH = sqrt(A_val * H)

    nX = (N1 + 1)^2
    nTvar = N * (N1 + 1)
    nvar = nX + nTvar

    # Variable bounds: default [0, 3.5], boundary X vars have uvar=10
    lvar = fill(T(0), nvar)
    uvar = fill(T(3.5), nvar)
    # T vars have default [0, 3.5]
    # Boundary X vars: X(I,0), X(I,N1), X(0,I), X(N1,I) → uvar=10
    for I in 1:N
        uvar[1 + I*(N1+1) + 0] = T(10)
        uvar[1 + I*(N1+1) + N1] = T(10)
        uvar[1 + 0*(N1+1) + I] = T(10)
        uvar[1 + N1*(N1+1) + I] = T(10)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar)

    # Objective: sum of T(I,J) variables
    ExaModels.@add_obj(c, x[i] for i in nX+1:nvar)

    # P(I,J) equality constraints for I=1..N, J=1..N
    # 4*X(I,J) - X(I,J+1) - X(I,J-1) - X(I+1,J) - X(I-1,J) = G*H^2
    p_data = [(c_ij   = 1 + I*(N1+1) + J,
               c_ijp  = 1 + I*(N1+1) + (J+1),
               c_ijm  = 1 + I*(N1+1) + (J-1),
               c_ip_j = 1 + (I+1)*(N1+1) + J,
               c_im_j = 1 + (I-1)*(N1+1) + J)
              for I in 1:N for J in 1:N]
    ExaModels.@add_con(c,
        T(4)*x[d.c_ij] - x[d.c_ijp] - x[d.c_ijm] - x[d.c_ip_j] - x[d.c_im_j] - GH2
        for d in p_data)

    # A/B constraints for I=1..N, J=1..N
    # A(I,J) G type: T(I,J) + H*X(I,J) >= RHS_A(I,J)
    # B(I,J) L type: -T(I,J) + H*X(I,J) <= RHS_B(I,J)
    # RHS_A = -5*I*H*(I*H-1)*J*H*(J*H-1)*H, RHS_B = +same magnitude
    ab_data = [(c_xij = 1 + I*(N1+1) + J,
                c_tij = nX + 1 + (I-1)*(N1+1) + J,
                rhs_a = T(-5)*T(I)*H*(T(I)*H-1)*T(J)*H*(T(J)*H-1)*H,
                rhs_b = T(5)*T(I)*H*(T(I)*H-1)*T(J)*H*(T(J)*H-1)*H)
               for I in 1:N for J in 1:N]
    ExaModels.@add_con(c,
        x[d.c_tij] + H*x[d.c_xij] - d.rhs_a for d in ab_data;
        lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c,
        -x[d.c_tij] + H*x[d.c_xij] - d.rhs_b for d in ab_data;
        lcon = T(-Inf), ucon = T(0))

    # Boundary constraints using T(I,0) (the J=0 T variable for each I)
    # C(I,0): T(I,0) + SQRTAH*X(I,0) >= 0
    # D(I,0): -T(I,0) + SQRTAH*X(I,0) <= 0
    cI0_data = [(c_x = 1 + I*(N1+1) + 0,
                 c_t = nX + 1 + (I-1)*(N1+1) + 0) for I in 1:N]
    ExaModels.@add_con(c, x[d.c_t] + SQRTAH*x[d.c_x] for d in cI0_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[d.c_t] + SQRTAH*x[d.c_x] for d in cI0_data;
                       lcon = T(-Inf), ucon = T(0))

    # C(I,N1): T(I,0) + SQRTAH*X(I,N1) >= 0  (uses T(I,0) per SIF!)
    # D(I,N1): -T(I,0) + SQRTAH*X(I,N1) <= 0
    cIN1_data = [(c_x = 1 + I*(N1+1) + N1,
                  c_t = nX + 1 + (I-1)*(N1+1) + 0) for I in 1:N]
    ExaModels.@add_con(c, x[d.c_t] + SQRTAH*x[d.c_x] for d in cIN1_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[d.c_t] + SQRTAH*x[d.c_x] for d in cIN1_data;
                       lcon = T(-Inf), ucon = T(0))

    # E(0,I): T(I,0) + SQRTAH*X(0,I) >= 0  (I=1..N)
    # F(0,I): -T(I,0) + SQRTAH*X(0,I) <= 0
    e0I_data = [(c_x = 1 + 0*(N1+1) + I,
                 c_t = nX + 1 + (I-1)*(N1+1) + 0) for I in 1:N]
    ExaModels.@add_con(c, x[d.c_t] + SQRTAH*x[d.c_x] for d in e0I_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[d.c_t] + SQRTAH*x[d.c_x] for d in e0I_data;
                       lcon = T(-Inf), ucon = T(0))

    # E(N1,I): T(I,0) + SQRTAH*X(N1,I) >= 0
    # F(N1,I): -T(I,0) + SQRTAH*X(N1,I) <= 0
    eN1I_data = [(c_x = 1 + N1*(N1+1) + I,
                  c_t = nX + 1 + (I-1)*(N1+1) + 0) for I in 1:N]
    ExaModels.@add_con(c, x[d.c_t] + SQRTAH*x[d.c_x] for d in eN1I_data;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[d.c_t] + SQRTAH*x[d.c_x] for d in eN1I_data;
                       lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end

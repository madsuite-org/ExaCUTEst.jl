# SPECAN: Spectral analysis problem (sum of K Gaussians fit to M observation points)
# K=3 Gaussians, each with 3 parameters (amplitude, center, width)
# M=5000 observation points at T_i = 1 + i/200 for i=1..M
# Objective: sum_{p=1..K} sum_{I=1..M} 0.5*(X(p,1)*exp(-(T_I-X(p,2))^2/X(p,3)^2) - Y_{p,I})^2
function specan_model(; T = Float64, backend = nothing, kwargs...)
    K = 3  # number of Gaussians
    N = 3  # variables per Gaussian
    M = 5000

    # Solution parameters (used to generate observation targets)
    SOLN = T[19.0 4.2 1.2; 8.0 2.5 4.6; 10.0 2.0 2.6]

    # H = 25.0 / M  (RD H RealM 25.0)
    H = T(25) / T(M)

    # Observation times T_I = 1 + H*I
    TI_vec = [T(1) + H * T(i) for i in 1:M]

    # Compute Y_{p,I} = SOLN[p,1]*exp(-(T_I - SOLN[p,2])^2 / SOLN[p,3]^2)
    Y = Matrix{T}(undef, K, M)
    for p in 1:K
        for i in 1:M
            ti = TI_vec[i]
            diff = ti - SOLN[p, 2]
            Y[p, i] = SOLN[p, 1] * exp(-(diff^2) / (SOLN[p, 3]^2))
        end
    end

    # Variable bounds and starting point
    LOWER = T[15.0 3.5 0.3; 5.0 2.2 2.6; 5.0 1.2 1.3]
    UPPER = T[31.0 6.3 3.7; 15.0 5.3 6.2; 14.0 3.3 2.8]
    START = T[25.0 5.2 3.2; 7.0 4.1 3.6; 11.6 1.9 2.2]

    # Variables: X(p,j) -> (p-1)*N + j (p outer, j inner)
    lvar_vec = vec(transpose(LOWER))  # X(1,1),X(1,2),X(1,3), X(2,1),...
    uvar_vec = vec(transpose(UPPER))
    start_vec = vec(transpose(START))

    # Variable index: X(p,j) -> (p-1)*N + j
    # (DO p outer, DO j inner means X(1,1),X(1,2),X(1,3),X(2,1),...,X(K,N))

    c = ExaModels.ExaCore(T; backend = backend)
    ExaModels.@add_var(c, x, K * N; lvar = lvar_vec, uvar = uvar_vec, start = start_vec)

    # Add objective terms separately for each Gaussian p to avoid indexing issues
    # For p=1: variables x[1], x[2], x[3]
    TI1 = [TI_vec[i] for i in 1:M]
    Y1  = [Y[1, i]   for i in 1:M]
    ExaModels.@add_par(c, T1obs, TI1)
    ExaModels.@add_par(c, Y1obs, Y1)
    ExaModels.@add_obj(
        c,
        T(0.5) * (x[1] * exp(-(T1obs[i] - x[2])^2 / x[3]^2) - Y1obs[i])^2 for i in 1:M
    )

    # For p=2: variables x[4], x[5], x[6]
    TI2 = [TI_vec[i] for i in 1:M]
    Y2  = [Y[2, i]   for i in 1:M]
    ExaModels.@add_par(c, T2obs, TI2)
    ExaModels.@add_par(c, Y2obs, Y2)
    ExaModels.@add_obj(
        c,
        T(0.5) * (x[4] * exp(-(T2obs[i] - x[5])^2 / x[6]^2) - Y2obs[i])^2 for i in 1:M
    )

    # For p=3: variables x[7], x[8], x[9]
    TI3 = [TI_vec[i] for i in 1:M]
    Y3  = [Y[3, i]   for i in 1:M]
    ExaModels.@add_par(c, T3obs, TI3)
    ExaModels.@add_par(c, Y3obs, Y3)
    ExaModels.@add_obj(
        c,
        T(0.5) * (x[7] * exp(-(T3obs[i] - x[8])^2 / x[9]^2) - Y3obs[i])^2 for i in 1:M
    )

    return ExaModels.ExaModel(c; kwargs...)
end
